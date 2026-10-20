"""
==============================================================================
SentinelSpark — Spark ML Feature Pipeline Construction
==============================================================================
Constructs robust, leak-free Spark ML Pipelines incorporating StringIndexer,
OneHotEncoder, VectorAssembler, and StandardScaler with strict target exclusion.
==============================================================================
"""

from typing import Any, Dict, List, Optional, Tuple
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame

from src.utils import logger


class FeaturePipelineBuilder:
    """Constructs Spark ML transformation pipelines for network flow modeling.

    Guarantees strict target leakage isolation: targets ('label', 'attack_cat'),
    surrogate IDs ('id'), and runtime evaluation columns are explicitly blocked
    from entering the assembled feature vector.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feat_cfg = config.get("features", {})
        self.prep_cfg = config.get("preprocessing", {})

        self.categorical_cols: List[str] = self.feat_cfg.get(
            "categorical_cols", ["proto", "service", "state"]
        )
        self.continuous_cols: List[str] = self.feat_cfg.get(
            "continuous_numeric_cols", []
        )
        self.ratio_cols: List[str] = self.feat_cfg.get("engineered_ratio_cols", [])
        self.binary_indicator_cols: List[str] = self.feat_cfg.get(
            "binary_indicator_cols", ["is_ftp_login", "is_sm_ips_ports"]
        )
        self.standardize_features: bool = self.prep_cfg.get("standardize_features", True)

    def get_feature_columns(
        self,
        df_columns: Optional[List[str]] = None,
        engineered_cols: Optional[List[str]] = None,
    ) -> List[str]:
        """Resolve and validate the complete list of numeric features to assemble into the vector."""
        if engineered_cols is None:
            engineered_cols = self.ratio_cols

        # Combine all numeric candidate columns
        candidate_numeric = list(
            dict.fromkeys(self.continuous_cols + engineered_cols + self.binary_indicator_cols)
        )

        # Strict leakage protection: block targets, IDs, and prediction columns
        forbidden = {
            "id",
            "index",
            "_c0",
            "_c1",
            "label",
            "attack_cat",
            "indexed_label",
            "raw_features",
            "features",
            "rawPrediction",
            "probability",
            "prediction",
        }
        forbidden.update(self.categorical_cols)

        numeric_features = [c for c in candidate_numeric if c not in forbidden]

        # If a reference DataFrame column list is provided, keep only available columns
        if df_columns is not None:
            numeric_features = [c for c in numeric_features if c in df_columns]

        return numeric_features

    def build_pipeline(
        self,
        target_mode: str = "binary",
        engineered_cols: Optional[List[str]] = None,
        include_multiclass_target_indexer: Optional[bool] = None,
        df_columns: Optional[List[str]] = None,
    ) -> Tuple[Pipeline, List[str]]:
        """Construct reusable Spark ML Pipeline stages with target leakage protection.

        Args:
            target_mode: 'binary' or 'multiclass'
            engineered_cols: Optional list of additional engineered feature columns
            include_multiclass_target_indexer: Explicit toggle for indexing attack_cat
            df_columns: Optional schema column names for automatic availability checking

        Returns:
            Tuple of (Pipeline, assembled_feature_input_names)
        """
        stages = []
        indexed_cat_cols = [f"{col}_idx" for col in self.categorical_cols]
        encoded_cat_cols = [f"{col}_vec" for col in self.categorical_cols]

        # 1. StringIndexer for categorical features (handles unknown tokens gracefully)
        handle_invalid = self.prep_cfg.get("string_indexer_handle_invalid", "keep")
        indexer = StringIndexer(
            inputCols=self.categorical_cols,
            outputCols=indexed_cat_cols,
            handleInvalid=handle_invalid,
        )
        stages.append(indexer)

        # 2. OneHotEncoder for indexed categorical variables
        encoder = OneHotEncoder(
            inputCols=indexed_cat_cols,
            outputCols=encoded_cat_cols,
            dropLast=True,
        )
        stages.append(encoder)

        # 3. Target Indexer for Stage 2 Multiclass profiling
        should_index_target = (
            include_multiclass_target_indexer
            if include_multiclass_target_indexer is not None
            else (target_mode == "multiclass")
        )

        if should_index_target:
            target_col = self.feat_cfg.get("target_multiclass", "attack_cat")
            target_indexer = StringIndexer(
                inputCol=target_col,
                outputCol="indexed_label",
                handleInvalid="keep",
            )
            stages.append(target_indexer)

        # 4. Resolve assembled numeric features with strict leakage exclusion
        numeric_features = self.get_feature_columns(
            df_columns=df_columns, engineered_cols=engineered_cols
        )
        assembler_inputs = encoded_cat_cols + numeric_features

        # 5. VectorAssembler to group numeric and one-hot categorical features
        raw_output_col = "raw_features" if self.standardize_features else "features"
        assembler = VectorAssembler(
            inputCols=assembler_inputs,
            outputCol=raw_output_col,
            handleInvalid="keep",
        )
        stages.append(assembler)

        # 6. StandardScaler (configurable) for feature normalization
        if self.standardize_features:
            scaler = StandardScaler(
                inputCol="raw_features",
                outputCol="features",
                withMean=False,
                withStd=True,
            )
            stages.append(scaler)

        pipeline = Pipeline(stages=stages)
        logger.info(
            f"Built ML Pipeline for [{target_mode}] with {len(stages)} stages "
            f"({len(assembler_inputs)} assembled input features)."
        )
        return pipeline, assembler_inputs

    def fit(
        self,
        df: DataFrame,
        target_mode: str = "binary",
        engineered_cols: Optional[List[str]] = None,
        include_multiclass_target_indexer: bool = False,
    ) -> PipelineModel:
        """Build and fit the pipeline on a DataFrame, returning a fitted PipelineModel."""
        pipeline, _ = self.build_pipeline(
            target_mode=target_mode,
            engineered_cols=engineered_cols,
            include_multiclass_target_indexer=include_multiclass_target_indexer,
            df_columns=df.columns,
        )
        logger.info(f"Fitting Feature Pipeline on DataFrame ({df.count():,} records)...")
        pipeline_model = pipeline.fit(df)
        return pipeline_model
