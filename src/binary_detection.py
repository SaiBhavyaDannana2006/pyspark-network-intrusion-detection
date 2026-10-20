"""
==============================================================================
SentinelSpark — Stage 1: Binary Intrusion Detection Engine
==============================================================================
Trains and evaluates binary classification models (Logistic Regression,
Random Forest, Gradient-Boosted Trees) on network traffic. Selects the
optimal model prioritizing Attack Recall and F1-score to maximize threat
detection and minimize false negatives.
==============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pyspark.ml.classification import (
    GBTClassificationModel,
    GBTClassifier,
    LogisticRegression,
    LogisticRegressionModel,
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.utils import logger


@dataclass
class BinaryDetectionResult:
    """Encapsulates Stage 1 training results, model artifacts, and evaluation metrics."""

    best_model: Any
    best_model_name: str
    all_models: Dict[str, Any] = field(default_factory=dict)
    all_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)


def compute_binary_class_weights(
    df: DataFrame, label_col: str = "label", weight_col: str = "class_weight"
) -> DataFrame:
    """Compute balanced inverse-frequency class weights for binary classification.

    Weight = total_records / (2 * class_records). Prioritizes rare attack or normal traffic.
    """
    counts = {r[label_col]: r["count"] for r in df.groupBy(label_col).count().collect()}
    total = sum(counts.values())
    if total == 0:
        return df.withColumn(weight_col, F.lit(1.0))
    n0 = counts.get(0.0, counts.get(0, 1))
    n1 = counts.get(1.0, counts.get(1, 1))
    w0 = float(total / (2.0 * max(n0, 1)))
    w1 = float(total / (2.0 * max(n1, 1)))
    logger.info(
        f"Stage 1 Class Balance Weights -> Normal (0): {w0:.4f} (count: {n0}), "
        f"Attack (1): {w1:.4f} (count: {n1})"
    )
    return df.withColumn(
        weight_col,
        F.when(F.col(label_col) == 1, F.lit(w1)).otherwise(F.lit(w0)),
    )


class BinaryDetectionFactory:
    """Instantiates configured Stage 1 binary classification estimators."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_cfg = config.get("models", {}).get("stage1_binary", {})

    def get_logistic_regression(self, use_weights: bool = False) -> LogisticRegression:
        """Create configured LogisticRegression estimator with class weighting."""
        lr_cfg = self.models_cfg.get("logistic_regression", {})
        weight_col = lr_cfg.get("weight_col", "class_weight") if use_weights else None
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=lr_cfg.get("max_iter", 50),
            regParam=lr_cfg.get("reg_param", 0.01),
            elasticNetParam=lr_cfg.get("elastic_net_param", 0.0),
            standardization=lr_cfg.get("standardization", True),
        )
        if weight_col:
            lr.setWeightCol(weight_col)
        return lr

    def get_random_forest(self, use_weights: bool = False) -> RandomForestClassifier:
        """Create configured RandomForestClassifier estimator with maxBins=128 and class weighting."""
        rf_cfg = self.models_cfg.get("random_forest", {})
        weight_col = rf_cfg.get("weight_col", "class_weight") if use_weights else None
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=rf_cfg.get("num_trees", 100),
            maxDepth=rf_cfg.get("max_depth", 12),
            maxBins=rf_cfg.get("max_bins", 128),
            featureSubsetStrategy=rf_cfg.get("feature_subset_strategy", "auto"),
            seed=rf_cfg.get("seed", 42),
        )
        if weight_col:
            rf.setWeightCol(weight_col)
        return rf

    def get_gbt(self) -> GBTClassifier:
        """Create configured GBTClassifier with maxBins=128 and deeper trees."""
        gbt_cfg = self.models_cfg.get("gbt", {})
        return GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=gbt_cfg.get("max_iter", 40),
            maxDepth=gbt_cfg.get("max_depth", 8),
            maxBins=gbt_cfg.get("max_bins", 128),
            stepSize=gbt_cfg.get("step_size", 0.1),
            seed=gbt_cfg.get("seed", 42),
        )


class BinaryDetectionEngine:
    """Stage 1 Binary Detection Engine comparing LR, Random Forest, and GBT classifiers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.factory = BinaryDetectionFactory(config)

    def evaluate_model(
        self, predictions: DataFrame, model_name: str
    ) -> Dict[str, float]:
        """Compute performance metrics emphasizing intrusion attack recall and F1-score."""
        total_records = predictions.count()

        # Compute Attack Recall (label == 1.0)
        attack_records = predictions.filter(F.col("label") == 1.0)
        total_attacks = attack_records.count()
        true_positives = attack_records.filter(F.col("prediction") == 1.0).count()
        attack_recall = float(true_positives / total_attacks) if total_attacks > 0 else 0.0

        # Standard Multi/Binary Evaluators
        f1_eval = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        )
        f1_score = float(f1_eval.evaluate(predictions))

        acc_eval = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        accuracy = float(acc_eval.evaluate(predictions))

        roc_auc = 0.5
        try:
            roc_eval = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
            )
            roc_auc = float(roc_eval.evaluate(predictions))
        except Exception:
            pass

        # Weighted selection score prioritizing intrusion recall (60%) over F1 (40%)
        selection_score = round((0.60 * attack_recall) + (0.40 * f1_score), 4)

        metrics = {
            "attack_recall": round(attack_recall, 4),
            "f1_score": round(f1_score, 4),
            "selection_score": selection_score,
            "accuracy": round(accuracy, 4),
            "roc_auc": round(roc_auc, 4),
            "true_positives": true_positives,
            "total_attacks": total_attacks,
            "total_records": total_records,
        }
        logger.info(
            f"Stage 1 [{model_name}] - Recall: {attack_recall:.4f} | F1: {f1_score:.4f} | "
            f"Accuracy: {accuracy:.4f} | Selection Score: {selection_score:.4f}"
        )
        return metrics

    def train_and_evaluate_all(
        self, train_df: DataFrame, val_df: DataFrame
    ) -> BinaryDetectionResult:
        """Train all 3 candidate models on train_df and select best based on val_df attack recall/F1."""
        logger.info("Executing Stage 1 binary classifier comparative training with class balancing...")

        # Calculate class weights on training split to boost rare class accuracy & recall
        weighted_train_df = compute_binary_class_weights(train_df, label_col="label", weight_col="class_weight")

        candidates = {
            "LogisticRegression": self.factory.get_logistic_regression(use_weights=True),
            "RandomForestClassifier": self.factory.get_random_forest(use_weights=True),
            "GBTClassifier": self.factory.get_gbt(),
        }

        all_models = {}
        all_metrics = {}
        best_model = None
        best_model_name = ""
        highest_score = -1.0

        for name, estimator in candidates.items():
            logger.info(f"Fitting candidate Stage 1 model: {name} (maxBins=128)...")
            # If estimator has weightCol set, use weighted_train_df; otherwise unweighted
            has_weight = estimator.isDefined("weightCol") and bool(estimator.getOrDefault("weightCol"))
            fit_df = weighted_train_df if has_weight else train_df
            fitted_model = estimator.fit(fit_df)
            all_models[name] = fitted_model

            # Predict probabilities and binary labels on validation set
            val_preds = fitted_model.transform(val_df)
            metrics = self.evaluate_model(val_preds, name)
            all_metrics[name] = metrics

            if metrics["selection_score"] > highest_score:
                highest_score = metrics["selection_score"]
                best_model = fitted_model
                best_model_name = name

        logger.info(
            f"Stage 1 Selection Winner: {best_model_name} with selection score: {highest_score:.4f}"
        )

        return BinaryDetectionResult(
            best_model=best_model,
            best_model_name=best_model_name,
            all_models=all_models,
            all_metrics=all_metrics,
            best_metrics=all_metrics[best_model_name],
        )


class BinaryDetector:
    """Backward-compatible estimator factory for legacy callers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.factory = BinaryDetectionFactory(config)

    def get_models(self) -> Dict[str, Any]:
        """Instantiate Stage 1 classifiers."""
        return {
            "logistic_regression": self.factory.get_logistic_regression(),
            "random_forest": self.factory.get_random_forest(),
            "gbt": self.factory.get_gbt(),
        }
