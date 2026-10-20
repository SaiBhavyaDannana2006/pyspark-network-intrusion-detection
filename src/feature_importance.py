"""
==============================================================================
SentinelSpark — Feature Importance Extraction & Vector Mapping Engine
==============================================================================
Extracts, ranks, and maps Gini feature importances from tree ensembles
(Random Forest, GBT) back to human-readable network domain feature names.
==============================================================================
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pyspark.ml import PipelineModel

from src.utils import logger


class FeatureImportanceExtractor:
    """Extracts and maps tree feature importances back to domain column names."""

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []

    def extract_tree_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Extract feature importances from a fitted RandomForest or GBT model or PipelineModel."""
        names = feature_names if feature_names is not None else self.feature_names

        # 1. Resolve featureImportances or coefficients from model or within pipeline stages
        if hasattr(model, "featureImportances"):
            raw_importances = model.featureImportances.toArray()
        elif hasattr(model, "coefficients"):
            raw_importances = np.abs(model.coefficients.toArray())
        elif hasattr(model, "stages"):
            found = False
            for stage in reversed(model.stages):
                if hasattr(stage, "featureImportances"):
                    raw_importances = stage.featureImportances.toArray()
                    found = True
                    break
                elif hasattr(stage, "coefficients"):
                    raw_importances = np.abs(stage.coefficients.toArray())
                    found = True
                    break
            if not found:
                raise AttributeError("No stage containing 'featureImportances' or 'coefficients' was discovered.")
        else:
            raise AttributeError(f"Model {type(model)} does not possess 'featureImportances' or 'coefficients'.")

        num_features = len(raw_importances)

        # 2. Map indices to resolved feature names
        if names and len(names) >= num_features:
            resolved_names = names[:num_features]
        elif names:
            resolved_names = list(names) + [f"feature_{i}" for i in range(len(names), num_features)]
        else:
            resolved_names = [f"feature_{i}" for i in range(num_features)]

        # 3. Assemble and rank
        df = pd.DataFrame({
            "feature": resolved_names,
            "importance": [round(float(v), 6) for v in raw_importances],
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        logger.info(f"Extracted feature importances for {len(df)} features. Top: {df.iloc[0]['feature']} ({df.iloc[0]['importance']:.4f})")
        return df

    def get_top_n(self, importance_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Filter and return the top N most influential features."""
        return importance_df.head(n).copy().reset_index(drop=True)


# Backward-compatible alias
FeatureImportanceAnalyzer = FeatureImportanceExtractor
