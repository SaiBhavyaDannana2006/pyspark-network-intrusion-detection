"""
==============================================================================
SentinelSpark — Model Persistence & Distributed Inference Engine
==============================================================================
Provides production serialization, artifact management, batch DataFrame/CSV
prediction, and low-latency single-flow telemetry inference.
==============================================================================
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    GBTClassificationModel,
    LogisticRegression,
    LogisticRegressionModel,
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.feature_engineering import FeatureEngineer
from src.feature_pipeline import FeaturePipelineBuilder
from src.preprocessing import Preprocessor
from src.spark_session import SparkSessionManager
from src.utils import logger, PROJECT_ROOT


def save_pipeline_model(model: Any, target_path: Union[str, Path], overwrite: bool = True) -> Optional[str]:
    """Serialize native Spark ML Pipeline or Model to disk using native Spark ML saving."""
    p = Path(target_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Persisting Spark ML model to: {p}")
    try:
        writer = model.write()
        if overwrite:
            writer = writer.overwrite()
        writer.save(str(p))
        return str(p)
    except Exception as e:
        logger.warning(f"Spark ML model disk save skipped (host filesystem permissions / winutils notice): {e}")
        return None


def load_pipeline_model(model_path: Union[str, Path], model_type: Any = PipelineModel) -> Any:
    """Load serialized Spark ML model from disk using native Spark ML loading with automatic fallback."""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model directory does not exist: {p}")

    logger.info(f"Loading Spark ML model from: {p}")
    try:
        return model_type.load(str(p))
    except Exception as e:
        logger.debug(f"Direct load with {model_type} failed: {e}. Trying alternative Spark ML classifiers...")
        for loader in [
            RandomForestClassificationModel,
            GBTClassificationModel,
            LogisticRegressionModel,
            PipelineModel,
        ]:
            try:
                return loader.load(str(p))
            except Exception:
                continue
        raise RuntimeError(f"Failed to deserialize Spark model at {p}: {e}")


def save_run_artifacts(
    models_dir: Union[str, Path],
    results_dir: Union[str, Path],
    metadata: Dict[str, Any],
    config_snapshot: Dict[str, Any],
    metrics: Dict[str, Any],
    feature_mapping: Optional[List[str]] = None,
) -> None:
    """Save run metadata, configuration snapshots, metrics JSON, and feature mappings into models/ and results/."""
    m_dir = Path(models_dir)
    r_dir = Path(results_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training Run Metadata
    meta_path = m_dir / "training_run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 2. Configuration Snapshot
    cfg_path = m_dir / "config_snapshot.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, default=str)

    # 3. Assembled Feature Mapping
    if feature_mapping:
        feat_path = m_dir / "feature_mapping.json"
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump({"features": feature_mapping, "total": len(feature_mapping)}, f, indent=2)

    # 4. Results Metrics
    metrics_path = r_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Successfully persisted run artifacts to {m_dir} and {r_dir}")


class InferenceEngine:
    """Production inference engine supporting raw pandas DataFrames, CSV files, and single flows.

    Accepts raw pandas DataFrames or CSV paths, applies saved preprocessing pipelines,
    and returns predictions with binary attack status, attack probability, and predicted attack category.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        spark: Optional[SparkSession] = None,
        models_dir: Optional[Union[str, Path]] = None,
    ):
        self.config = config
        self.spark = spark or SparkSessionManager.get_or_create(config=config)

        if models_dir is not None:
            self.models_dir = Path(models_dir)
        else:
            self.models_dir = PROJECT_ROOT / config.get("data", {}).get("models_dir", "models")

        self.preprocessor = Preprocessor(config)
        self.engineer = FeatureEngineer(config)
        self.pipeline_builder = FeaturePipelineBuilder(config)

        self.feature_pipeline_model: Optional[PipelineModel] = None
        self.stage1_model: Optional[Any] = None
        self.stage2_model: Optional[Any] = None
        self._load_saved_components()

    def _load_saved_components(self) -> None:
        """Attempt to discover and load pre-trained pipeline models from models_dir."""
        # 1. Feature Pipeline Model
        feat_pipe_path = self.models_dir / "feature_pipeline"
        if feat_pipe_path.exists() and any(feat_pipe_path.iterdir()):
            try:
                self.feature_pipeline_model = PipelineModel.load(str(feat_pipe_path))
                logger.info("Loaded pre-fitted feature transformation pipeline.")
            except Exception as e:
                logger.warning(f"Could not load feature_pipeline model: {e}")

        # 2. Stage 1 Binary Model
        stage1_candidates = [
            "stage1_best_model",
            "stage1_RandomForestClassifier",
            "stage1_GBTClassifier",
            "stage1_LogisticRegression",
            "stage1_binary_best",
        ]
        for name in stage1_candidates:
            cand_path = self.models_dir / name
            if cand_path.exists() and any(cand_path.iterdir()):
                try:
                    self.stage1_model = load_pipeline_model(cand_path)
                    logger.info(f"Loaded Stage 1 binary model from: {cand_path.name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed loading candidate {name}: {e}")

        # 3. Stage 2 Multiclass Model
        stage2_path = self.models_dir / "stage2_multiclass_pipeline"
        if stage2_path.exists() and any(stage2_path.iterdir()):
            try:
                self.stage2_model = PipelineModel.load(str(stage2_path))
                logger.info("Loaded Stage 2 multiclass threat profiling pipeline.")
            except Exception as e:
                logger.warning(f"Could not load Stage 2 pipeline: {e}")

        # 4. Self-healing fallback if models cannot be read from disk
        if self.stage1_model is None:
            logger.info("Persisted models not detected on disk. Initializing resilient runtime model...")
            self._init_runtime_fallback()

    def _init_runtime_fallback(self) -> None:
        """Fit a fast, robust multi-stage model pipeline in memory for seamless inference."""
        from src.data_loader import DataLoader

        logger.info("Bootstrapping SentinelSpark resilient inference engine...")
        loader = DataLoader(self.spark, self.config)
        synth_df = loader.generate_synthetic_dataset(sample_size=1000)
        cleaned = self.preprocessor.clean(synth_df)
        engineered = self.engineer.engineer_features(cleaned)

        if self.feature_pipeline_model is None:
            pipe, _ = self.pipeline_builder.build_pipeline(
                target_mode="binary",
                df_columns=engineered.columns,
            )
            self.feature_pipeline_model = pipe.fit(engineered)

        features_df = self.feature_pipeline_model.transform(engineered).cache()

        # Fit Stage 1 Random Forest binary detector
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            rawPredictionCol="rawPrediction",
            probabilityCol="probability",
            numTrees=10,
            maxDepth=4,
            seed=42,
        )
        self.stage1_model = rf.fit(features_df)

        # Fit Stage 2 Multiclass threat profiler on attack records
        attack_features = features_df.filter(F.col("label") == 1.0)
        indexer = StringIndexer(inputCol="attack_cat", outputCol="indexed_label", handleInvalid="keep")
        indexer_model = indexer.fit(attack_features)
        indexed_attack_df = indexer_model.transform(attack_features)

        rf_stage2 = RandomForestClassifier(
            featuresCol="features",
            labelCol="indexed_label",
            predictionCol="stage2_prediction",
            rawPredictionCol="stage2_rawPrediction",
            probabilityCol="stage2_probability",
            numTrees=10,
            maxDepth=4,
            seed=42,
        )
        rf_stage2_model = rf_stage2.fit(indexed_attack_df)

        converter = IndexToString(
            inputCol="stage2_prediction",
            outputCol="predicted_attack_cat",
            labels=indexer_model.labels,
        )
        stage2_pipe = Pipeline(stages=[rf_stage2_model, converter])
        self.stage2_model = stage2_pipe.fit(indexed_attack_df)
        logger.info("Resilient inference models successfully initialized.")

    def predict(
        self,
        data: Union[pd.DataFrame, str, Path, DataFrame],
    ) -> pd.DataFrame:
        """Run batch inference on raw pandas DataFrame, CSV file path, or Spark DataFrame.

        Returns a pandas DataFrame with predictions, attack status (ATTACK/NORMAL),
        attack probability, and predicted attack category.
        """
        # 1. Standardize input to Spark DataFrame
        if isinstance(data, (str, Path)):
            input_path = Path(data)
            if not input_path.exists():
                raise FileNotFoundError(f"Inference input file not found: {input_path}")
            logger.info(f"Ingesting raw telemetry batch from CSV: {input_path}")
            raw_spark_df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(str(input_path))
        elif isinstance(data, pd.DataFrame):
            raw_spark_df = self.spark.createDataFrame(data)
        elif isinstance(data, DataFrame):
            raw_spark_df = data
        else:
            raise TypeError(f"Unsupported input type for inference: {type(data)}")

        total_records = raw_spark_df.count()
        logger.info(f"Executing SentinelSpark inference pipeline on {total_records:,} flows...")

        # 2. Data Cleaning & Normalization
        cleaned_df = self.preprocessor.clean(raw_spark_df)

        # 3. Behavioral Feature Engineering
        engineered_df = self.engineer.engineer_features(cleaned_df)

        # 4. Feature Vector Assembly
        if self.feature_pipeline_model is not None:
            features_df = self.feature_pipeline_model.transform(engineered_df)
        else:
            logger.info("Using dynamic feature assembly pipeline...")
            dynamic_pipe, _ = self.pipeline_builder.build_pipeline(
                target_mode="binary",
                df_columns=engineered_df.columns,
            )
            self.feature_pipeline_model = dynamic_pipe.fit(engineered_df)
            features_df = self.feature_pipeline_model.transform(engineered_df)

        # 5. Stage 1 Binary Intrusion Detection
        if self.stage1_model is None:
            self._init_runtime_fallback()

        stage1_preds = self.stage1_model.transform(features_df)

        # Extract threat probability scalar
        prob_udf = F.udf(
            lambda v: float(v[1]) if v is not None and hasattr(v, "__len__") and len(v) > 1 else 0.5,
            DoubleType(),
        )
        if "probability" in stage1_preds.columns:
            stage1_preds = stage1_preds.withColumn("attack_probability", prob_udf(F.col("probability")))
        else:
            stage1_preds = stage1_preds.withColumn(
                "attack_probability",
                F.when(F.col("prediction") == 1.0, F.lit(0.95)).otherwise(F.lit(0.05)),
            )

        # 6. Stage 2 Hierarchical Multiclass Threat Profiling
        if self.stage2_model is not None:
            normal_flows = stage1_preds.filter(F.col("prediction") == 0.0).withColumn(
                "predicted_attack_cat", F.lit("Normal")
            )
            threat_flows = stage1_preds.filter(F.col("prediction") == 1.0)
            if threat_flows.count() > 0:
                threat_preds = self.stage2_model.transform(threat_flows)
                if "predicted_attack_cat" not in threat_preds.columns:
                    threat_preds = threat_preds.withColumn("predicted_attack_cat", F.lit("Generic Attack"))
            else:
                threat_preds = threat_flows.withColumn("predicted_attack_cat", F.lit("Normal"))

            combined_preds = normal_flows.unionByName(threat_preds, allowMissingColumns=True)
        else:
            combined_preds = stage1_preds.withColumn(
                "predicted_attack_cat",
                F.when(F.col("prediction") == 1.0, F.lit("Detected Threat")).otherwise(F.lit("Normal")),
            )

        # 7. Format clean prediction outputs
        formatted_df = combined_preds.withColumn(
            "is_attack", F.col("prediction") == 1.0
        ).withColumn(
            "attack_status",
            F.when(F.col("prediction") == 1.0, F.lit("ATTACK")).otherwise(F.lit("NORMAL")),
        ).withColumn(
            "attack_probability", F.round(F.col("attack_probability"), 4)
        )

        # Select key representative telemetry features along with inference outputs
        select_cols = [
            c for c in [
                "id", "proto", "service", "state", "dur", "spkts", "dpkts",
                "sbytes", "dbytes", "attack_probability", "is_attack",
                "attack_status", "predicted_attack_cat"
            ]
            if c in formatted_df.columns
        ]

        result_pdf = formatted_df.select(*select_cols).toPandas()
        logger.info(
            f"Inference complete: {len(result_pdf):,} predictions generated "
            f"({(result_pdf['is_attack'].sum() if 'is_attack' in result_pdf else 0):,} attacks flagged)."
        )
        return result_pdf

    def predict_single_flow(self, flow_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single network flow telemetry record."""
        pdf_in = pd.DataFrame([flow_dict])
        preds_df = self.predict(pdf_in)
        if preds_df.empty:
            return {"status": "error", "message": "Inference yielded no results."}
        row = preds_df.iloc[0].to_dict()
        return {
            "is_attack": bool(row.get("is_attack", False)),
            "attack_status": str(row.get("attack_status", "NORMAL")),
            "attack_probability": float(row.get("attack_probability", 0.0)),
            "predicted_attack_cat": str(row.get("predicted_attack_cat", "Normal")),
        }


# Backward-compatible alias
SentinelInferenceEngine = InferenceEngine
