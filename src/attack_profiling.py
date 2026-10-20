"""
==============================================================================
SentinelSpark — Stage 2: Multiclass Attack Category Profiling
==============================================================================
Multiclass classification models for granular taxonomy categorization of
detected threats (e.g., DoS, Exploits, Reconnaissance, Fuzzers, Backdoor).
Features hierarchical routing: Stage 2 models evaluate only malicious flows
flagged by Stage 1 binary detection.
==============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.utils import logger


def compute_multiclass_weights(
    df: DataFrame,
    label_col: str = "indexed_attack_cat",
    weight_col: str = "class_weight",
) -> DataFrame:
    """Compute inverse class frequency weights across attack categories to balance rare attack classes."""
    counts = {r[label_col]: r["count"] for r in df.groupBy(label_col).count().collect()}
    total = sum(counts.values())
    num_classes = max(len(counts), 1)

    when_expr = None
    for cat_val, cat_count in counts.items():
        w = float(total / (num_classes * max(cat_count, 1)))
        cond = F.col(label_col) == cat_val
        if when_expr is None:
            when_expr = F.when(cond, F.lit(w))
        else:
            when_expr = when_expr.when(cond, F.lit(w))

    if when_expr is not None:
        return df.withColumn(weight_col, when_expr.otherwise(F.lit(1.0)))
    return df.withColumn(weight_col, F.lit(1.0))


@dataclass
class Stage2ProfilingResult:
    """Encapsulates Stage 2 multiclass model artifacts and category mappings."""

    pipeline_model: PipelineModel
    labels: List[str]
    training_metrics: Dict[str, float] = field(default_factory=dict)
    model_name: str = "MulticlassRandomForest"


class AttackProfilingEngine:
    """Stage 2 Threat Taxonomy Profiler with hierarchical pipeline routing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rf_cfg = config.get("models", {}).get("stage2_multiclass", {}).get("random_forest", {})

    def train(self, train_df: DataFrame) -> Stage2ProfilingResult:
        """Train Stage 2 Profiler strictly on malicious intrusion records (excluding Normal)."""
        logger.info("Training Stage 2 Attack Taxonomy Profiler strictly on threat records...")

        # 1. Filter out legitimate flows (label == 0 or attack_cat == 'Normal')
        threat_df = train_df.filter(
            (F.col("label") == 1)
            & (F.col("attack_cat") != "Normal")
            & (F.col("attack_cat").isNotNull())
        )

        threat_count = threat_df.count()
        logger.info(f"Filtered training set for Stage 2: {threat_count:,} malicious flows.")
        if threat_count == 0:
            raise ValueError("Cannot train Stage 2 Attack Profiler: No malicious records found in training split.")

        # 2. StringIndex attack_cat target column strictly on attacks
        target_indexer = StringIndexer(
            inputCol="attack_cat",
            outputCol="indexed_attack_cat",
            handleInvalid="keep",
        )
        indexer_model = target_indexer.fit(threat_df)
        labels = [str(label) for label in indexer_model.labels if label != "Normal"]

        logger.info(f"Stage 2 Profiler taxonomy classes ({len(labels)} categories): {labels}")

        threat_indexed = indexer_model.transform(threat_df)
        threat_weighted = compute_multiclass_weights(threat_indexed, label_col="indexed_attack_cat", weight_col="class_weight")

        # 3. Multiclass Random Forest Classifier with maxBins=128, deeper trees, and class weighting
        rf_classifier = RandomForestClassifier(
            featuresCol="features",
            labelCol="indexed_attack_cat",
            predictionCol="stage2_prediction",
            rawPredictionCol="stage2_rawPrediction",
            probabilityCol="stage2_probability",
            numTrees=self.rf_cfg.get("num_trees", 100),
            maxDepth=self.rf_cfg.get("max_depth", 14),
            maxBins=self.rf_cfg.get("max_bins", 128),
            weightCol="class_weight",
            featureSubsetStrategy=self.rf_cfg.get("feature_subset_strategy", "sqrt"),
            seed=self.rf_cfg.get("seed", 42),
        )
        rf_model = rf_classifier.fit(threat_weighted)

        # 4. Invert predicted indices back to taxonomy strings
        label_converter = IndexToString(
            inputCol="stage2_prediction",
            outputCol="predicted_attack_cat",
            labels=indexer_model.labels,
        )

        # 5. Assemble Stage 2 PipelineModel
        pipeline_model = PipelineModel(stages=[indexer_model, rf_model, label_converter])

        # Compute training metrics on malicious flows
        threat_preds = pipeline_model.transform(threat_df)
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="indexed_attack_cat",
            predictionCol="stage2_prediction",
            metricName="f1",
        )
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="indexed_attack_cat",
            predictionCol="stage2_prediction",
            metricName="accuracy",
        )

        train_f1 = float(evaluator_f1.evaluate(threat_preds))
        train_acc = float(evaluator_acc.evaluate(threat_preds))

        metrics = {
            "train_f1": round(train_f1, 4),
            "train_accuracy": round(train_acc, 4),
            "threat_sample_count": threat_count,
            "num_classes": len(labels),
        }
        logger.info(f"Stage 2 Attack Profiler trained successfully: {metrics}")

        return Stage2ProfilingResult(
            pipeline_model=pipeline_model,
            labels=labels,
            training_metrics=metrics,
            model_name="MulticlassRandomForest",
        )

    def evaluate_hierarchical(
        self,
        stage1_preds: DataFrame,
        stage2_result: Stage2ProfilingResult,
    ) -> Tuple[DataFrame, Dict[str, Any]]:
        """Hierarchical inference: evaluate Stage 2 taxonomy only on Stage 1 flagged threats.

        Flows flagged as legitimate (prediction == 0.0) are designated as 'Normal'.
        Flows flagged as malicious (prediction == 1.0) are routed through Stage 2.
        """
        logger.info("Executing hierarchical threat routing across Stage 1 predictions...")

        # 1. Isolate Normal flows (cleared by Stage 1)
        normal_flows = stage1_preds.filter(F.col("prediction") == 0.0).withColumn(
            "predicted_attack_cat", F.lit("Normal")
        )

        # 2. Isolate Threat flows (flagged by Stage 1)
        threat_flows = stage1_preds.filter(F.col("prediction") == 1.0)
        threat_count = threat_flows.count()

        if threat_count > 0:
            logger.info(f"Routing {threat_count:,} flagged threats through Stage 2 Profiler...")
            threat_preds = stage2_result.pipeline_model.transform(threat_flows)

            # Compute Stage 2 evaluation metrics on threats where ground truth attack_cat is present
            metrics: Dict[str, Any] = {"flagged_threats_count": threat_count}
            try:
                eval_subset = threat_preds.filter(
                    (F.col("attack_cat") != "Normal") & (F.col("attack_cat").isNotNull())
                )
                if eval_subset.count() > 0:
                    evaluator_f1 = MulticlassClassificationEvaluator(
                        labelCol="indexed_attack_cat",
                        predictionCol="stage2_prediction",
                        metricName="f1",
                    )
                    evaluator_acc = MulticlassClassificationEvaluator(
                        labelCol="indexed_attack_cat",
                        predictionCol="stage2_prediction",
                        metricName="accuracy",
                    )
                    metrics["stage2_f1"] = round(float(evaluator_f1.evaluate(eval_subset)), 4)
                    metrics["stage2_accuracy"] = round(float(evaluator_acc.evaluate(eval_subset)), 4)
            except Exception as e:
                logger.warning(f"Could not compute Stage 2 threat validation metrics: {e}")
        else:
            threat_preds = threat_flows.withColumn("predicted_attack_cat", F.lit("Normal"))
            metrics = {"flagged_threats_count": 0}

        # 3. Union predictions back together
        hierarchical_preds = normal_flows.unionByName(threat_preds, allowMissingColumns=True)

        return hierarchical_preds, metrics


class AttackProfiler:
    """Backward-compatible estimator factory for legacy callers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rf_cfg = config.get("models", {}).get("stage2_multiclass", {}).get("random_forest", {})

    def get_model(self) -> RandomForestClassifier:
        """Instantiate Stage 2 Multiclass Random Forest Classifier."""
        return RandomForestClassifier(
            featuresCol="features",
            labelCol="indexed_label",
            numTrees=self.rf_cfg.get("num_trees", 60),
            maxDepth=self.rf_cfg.get("max_depth", 12),
            featureSubsetStrategy=self.rf_cfg.get("feature_subset_strategy", "sqrt"),
            seed=self.rf_cfg.get("seed", 42),
        )
