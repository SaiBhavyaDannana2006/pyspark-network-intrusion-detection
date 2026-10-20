"""
==============================================================================
SentinelSpark — Model Training & Persistence Orchestrator
==============================================================================
Coordinates Stage 1 model comparison, hyperparameter tuning, best model selection,
Stage 2 multiclass threat taxonomy profiling, and hierarchical inference.
Serializes trained models and execution run metadata.
==============================================================================
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import DataFrame

from src.attack_profiling import AttackProfilingEngine, Stage2ProfilingResult
from src.binary_detection import BinaryDetectionEngine, BinaryDetectionResult
from src.hyperparameter_tuning import HyperparameterTuner
from src.utils import logger, PROJECT_ROOT, timer


@dataclass
class TrainingArtifacts:
    """Encapsulates end-to-end trained model artifacts, hierarchical predictions, and run metadata."""

    stage1_best_model: Any
    stage1_result: BinaryDetectionResult
    stage2_profiling: Stage2ProfilingResult
    hierarchical_val_predictions: DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelTrainer:
    """Orchestrates multi-stage ML model training, tuning, and pipeline serialization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        artifacts_dir_cfg = config.get("models", {}).get("artifacts_dir")
        if artifacts_dir_cfg:
            self.artifacts_dir = Path(artifacts_dir_cfg)
        else:
            self.artifacts_dir = PROJECT_ROOT / config.get("data", {}).get("models_dir", "models")

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.artifacts_dir

    def train_full_pipeline(
        self, train_df: DataFrame, val_df: DataFrame
    ) -> TrainingArtifacts:
        """Coordinate Stage 1 comparison, tuning, and Stage 2 profiling training."""
        logger.info("Initiating full multi-stage ML training pipeline orchestrator...")
        start_time = time.time()

        # 1. Stage 1 Binary Detection Comparative Evaluation
        binary_engine = BinaryDetectionEngine(self.config)
        stage1_result = binary_engine.train_and_evaluate_all(train_df, val_df)
        stage1_best_model = stage1_result.best_model

        # 2. Hyperparameter Tuning (if enabled)
        tuner = HyperparameterTuner(self.config)
        if tuner.is_enabled and isinstance(stage1_best_model, RandomForestClassificationModel):
            logger.info("Hyperparameter tuning enabled: Tuning Stage 1 Random Forest...")
            rf_estimator = binary_engine.factory.get_random_forest()
            tune_res = tuner.tune_random_forest(rf_estimator, train_df)
            stage1_best_model = tune_res.best_model

        # 3. Stage 2 Multiclass Attack Taxonomy Profiler
        profiling_engine = AttackProfilingEngine(self.config)
        stage2_result = profiling_engine.train(train_df)

        # 4. Hierarchical Prediction Routing on Validation Set
        stage1_val_preds = stage1_best_model.transform(val_df)
        hierarchical_preds, stage2_metrics = profiling_engine.evaluate_hierarchical(
            stage1_val_preds, stage2_result
        )

        # 5. Compile and Persist Pipeline Metadata
        duration_sec = round(time.time() - start_time, 2)
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "runtime_duration_seconds": duration_sec,
            "stage1": {
                "best_model_name": stage1_result.best_model_name,
                "best_metrics": stage1_result.best_metrics,
                "all_metrics": stage1_result.all_metrics,
            },
            "stage2": {
                "model_name": stage2_result.model_name,
                "taxonomy_labels": stage2_result.labels,
                "training_metrics": stage2_result.training_metrics,
                "hierarchical_metrics": stage2_metrics,
            },
        }

        # Save fitted models and pipeline metadata to disk
        try:
            self.save_model(stage1_best_model, f"stage1_{stage1_result.best_model_name}")
            self.save_model(stage2_result.pipeline_model, "stage2_multiclass_pipeline")
        except Exception as e:
            logger.warning(f"Non-critical model serialization notice: {e}")

        meta_file = self.artifacts_dir / "training_run_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Persisted pipeline execution metadata to: {meta_file}")

        return TrainingArtifacts(
            stage1_best_model=stage1_best_model,
            stage1_result=stage1_result,
            stage2_profiling=stage2_result,
            hierarchical_val_predictions=hierarchical_preds,
            metadata=metadata,
        )

    @timer
    def train(self, estimator: Any, train_df: DataFrame) -> Transformer:
        """Fit a single estimator on distributed training DataFrame."""
        logger.info(f"Fitting {estimator.__class__.__name__} on {train_df.count():,} records...")
        return estimator.fit(train_df)

    def save_model(self, model: Any, model_name: str, overwrite: bool = True) -> Path:
        """Serialize fitted Spark ML model or PipelineModel to disk."""
        target_path = self.artifacts_dir / model_name
        logger.info(f"Saving Spark ML model '{model_name}' to {target_path}...")

        if overwrite:
            model.write().overwrite().save(str(target_path))
        else:
            model.write().save(str(target_path))

        logger.info(f"Model successfully saved to {target_path}")
        return target_path

    def load_model(self, model_name: str, model_type: Any = PipelineModel) -> Optional[Any]:
        """Load serialized Spark ML model from disk."""
        target_path = self.artifacts_dir / model_name
        if not target_path.exists():
            logger.warning(f"Model path {target_path} does not exist.")
            return None
        logger.info(f"Loading Spark ML model from {target_path}...")
        return model_type.load(str(target_path))
