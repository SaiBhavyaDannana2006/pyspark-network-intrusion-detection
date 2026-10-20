"""
==============================================================================
SentinelSpark — Hyperparameter Tuning & Cross-Validation Engine
==============================================================================
Provides distributed hyperparameter optimization using ParamGridBuilder,
TrainValidationSplit, and CrossValidator. Features configurable fast-path
toggling to support resource-constrained demo environments.
==============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pyspark.ml.classification import RandomForestClassificationModel, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.sql import DataFrame

from src.utils import logger


@dataclass
class TuningResult:
    """Encapsulates hyperparameter tuning status, optimal estimator, and best parameters."""

    is_tuned: bool
    best_model: Any
    best_params: Dict[str, Any] = field(default_factory=dict)
    metric_value: Optional[float] = None


class HyperparameterTuner:
    """Manages distributed hyperparameter tuning and model selection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tuning_cfg: Dict[str, Any] = config.get("tuning", {})
        self.is_enabled: bool = self.tuning_cfg.get("enabled", False)

    def tune_random_forest(
        self,
        estimator: RandomForestClassifier,
        train_df: DataFrame,
    ) -> TuningResult:
        """Tune RandomForestClassifier using TrainValidationSplit with fast-path fallback."""
        # Ensure weightCol exists if configured on estimator
        if estimator.isDefined("weightCol"):
            w_col = estimator.getOrDefault("weightCol")
            if w_col and w_col not in train_df.columns and "label" in train_df.columns:
                from src.binary_detection import compute_binary_class_weights
                train_df = compute_binary_class_weights(train_df, weight_col=w_col)

        if not self.is_enabled:
            logger.info("Hyperparameter tuning is DISABLED (fast-path). Fitting base model...")
            fitted_model: RandomForestClassificationModel = estimator.fit(train_df)
            best_params = {
                "numTrees": fitted_model.getNumTrees,
                "maxDepth": fitted_model.getOrDefault("maxDepth"),
                "maxBins": fitted_model.getOrDefault("maxBins"),
            }
            return TuningResult(
                is_tuned=False,
                best_model=fitted_model,
                best_params=best_params,
            )

        logger.info("Hyperparameter tuning is ENABLED. Initializing ParamGridBuilder...")
        grids_cfg = self.tuning_cfg.get("param_grids", {})
        num_trees = grids_cfg.get("rf_num_trees", [50, 100])
        max_depth = grids_cfg.get("rf_max_depth", [8, 12])
        max_bins = grids_cfg.get("rf_max_bins", [64, 128])

        param_grid = (
            ParamGridBuilder()
            .addGrid(estimator.numTrees, [int(n) for n in num_trees])
            .addGrid(estimator.maxDepth, [int(d) for d in max_depth])
            .addGrid(estimator.maxBins, [int(b) for b in max_bins])
            .build()
        )

        logger.info(
            f"Searching parameter space ({len(param_grid)} parameter combinations): "
            f"numTrees={num_trees}, maxDepth={max_depth}, maxBins={max_bins}"
        )

        # Configurable metric selection: weightedF1 / areaUnderROC
        target_metric = self.tuning_cfg.get("metric_name", "weightedF1")
        if target_metric in ["areaUnderROC", "areaUnderPR"]:
            evaluator = BinaryClassificationEvaluator(
                rawPredictionCol="rawPrediction",
                labelCol="label",
                metricName=target_metric,
            )
        else:
            # PySpark MulticlassClassificationEvaluator uses 'f1' for weighted F1-measure
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="f1",
            )

        tvs = TrainValidationSplit(
            estimator=estimator,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            trainRatio=0.8,
            seed=self.config.get("splits", {}).get("seed", 42),
            parallelism=1,
        )

        tvs_model = tvs.fit(train_df)
        best_model: RandomForestClassificationModel = tvs_model.bestModel

        best_params = {
            "numTrees": best_model.getNumTrees,
            "maxDepth": best_model.getOrDefault("maxDepth"),
            "maxBins": best_model.getOrDefault("maxBins"),
        }
        best_metric = max(tvs_model.validationMetrics) if hasattr(tvs_model, "validationMetrics") else None
        metric_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"

        logger.info(
            f"Tuning complete! Optimal params: {best_params} ({target_metric}: {metric_str})"
        )

        return TuningResult(
            is_tuned=True,
            best_model=best_model,
            best_params=best_params,
            metric_value=best_metric,
        )

    def build_binary_cross_validator(
        self,
        estimator: Any,
        param_grid: Any,
        num_folds: int = 3,
        metric_name: str = "areaUnderROC",
    ) -> CrossValidator:
        """Create a CrossValidator instance for Stage 1 Binary models."""
        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol="label",
            metricName=metric_name,
        )
        logger.info(f"Building CrossValidator with {num_folds} folds optimizing for {metric_name}...")
        return CrossValidator(
            estimator=estimator,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
            seed=self.config.get("splits", {}).get("seed", 42),
            parallelism=2,
        )

    def build_multiclass_cross_validator(
        self,
        estimator: Any,
        param_grid: Any,
        num_folds: int = 3,
        metric_name: str = "f1",
    ) -> CrossValidator:
        """Create a CrossValidator instance for Stage 2 Multiclass models."""
        evaluator = MulticlassClassificationEvaluator(
            labelCol="indexed_label",
            predictionCol="prediction",
            metricName=metric_name,
        )
        logger.info(f"Building Multiclass CrossValidator with {num_folds} folds optimizing for {metric_name}...")
        return CrossValidator(
            estimator=estimator,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
            seed=self.config.get("splits", {}).get("seed", 42),
            parallelism=2,
        )
