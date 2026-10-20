"""
==============================================================================
SentinelSpark — Preprocessing, Data Cleaning & Leakage Protection
==============================================================================
Handles exact duplicate removal, zero-leakage median imputation (PySpark Imputer),
categorical null replacement, and training-fitted IQR outlier soft-capping.
==============================================================================
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml.feature import Imputer, ImputerModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.utils import logger


class Preprocessor:
    """Production data cleaning, imputation, and outlier treatment engine.

    Enforces strict zero-leakage guarantees by fitting all transformation
    parameters (medians, IQR thresholds) solely on training splits, then
    applying them immutably to evaluation and test partitions.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features_cfg = config.get("features", {})
        self.prep_cfg = config.get("preprocessing", {})
        self.splits_cfg = config.get("splits", {})

        self.categorical_cols: List[str] = self.features_cfg.get(
            "categorical_cols", ["proto", "service", "state"]
        )
        self.numeric_cols: List[str] = self.features_cfg.get(
            "continuous_numeric_cols", []
        )
        self.binary_indicator_cols: List[str] = self.features_cfg.get(
            "binary_indicator_cols", ["is_ftp_login", "is_sm_ips_ports"]
        )

        # Fitted parameters for zero-leakage propagation
        self.duplicate_count: int = 0
        self.imputer_model: Optional[ImputerModel] = None
        self.fitted_medians: Dict[str, float] = {}
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}

    def remove_duplicates(
        self, df: DataFrame, subset_cols: Optional[List[str]] = None
    ) -> DataFrame:
        """Remove exact duplicate network flow records and track deduplication delta."""
        before_count = df.count()

        if subset_cols is None:
            # Drop duplicates across flow features, excluding surrogate ID
            subset_cols = [c for c in df.columns if c.lower() not in ("id", "index", "_c0")]

        deduped_df = df.dropDuplicates(subset=subset_cols)
        after_count = deduped_df.count()
        self.duplicate_count = before_count - after_count

        logger.info(
            f"Deduplication complete: {before_count:,} -> {after_count:,} "
            f"(Removed {self.duplicate_count:,} exact duplicates)."
        )
        return deduped_df

    def handle_categoricals(self, df: DataFrame) -> DataFrame:
        """Strip whitespace, lowercase, and explicitly impute categorical nulls with 'unknown'."""
        cleaned_df = df

        for col_name in self.categorical_cols:
            if col_name in cleaned_df.columns:
                cleaned_df = cleaned_df.withColumn(
                    col_name,
                    F.when(
                        F.col(col_name).isNull()
                        | (F.trim(F.col(col_name)) == "")
                        | (F.trim(F.col(col_name)) == "null"),
                        F.lit("unknown"),
                    ).otherwise(F.lower(F.trim(F.col(col_name)))),
                )

        # Standardize attack_cat target column
        if "attack_cat" in cleaned_df.columns:
            cleaned_df = cleaned_df.withColumn(
                "attack_cat",
                F.when(
                    F.col("attack_cat").isNull()
                    | (F.trim(F.col("attack_cat")) == "")
                    | (F.trim(F.col("attack_cat")) == "-"),
                    F.lit("Normal"),
                ).otherwise(F.trim(F.col("attack_cat"))),
            )
            # Correct common UNSW-NB15 spelling discrepancies (e.g. 'Backdoors' -> 'Backdoor')
            cleaned_df = cleaned_df.withColumn(
                "attack_cat",
                F.when(F.col("attack_cat") == "Backdoors", F.lit("Backdoor"))
                .when(F.col("attack_cat") == "Fuzzers ", F.lit("Fuzzers"))
                .when(F.col("attack_cat") == "Reconnaissance ", F.lit("Reconnaissance"))
                .otherwise(F.col("attack_cat")),
            )

        return cleaned_df

    def fit_imputer(self, df: DataFrame) -> None:
        """Fit median values for continuous numeric columns using PySpark Imputer."""
        available_numeric = [
            c for c in self.numeric_cols
            if c in df.columns and c not in ("id", "label")
        ]

        if not available_numeric:
            return

        logger.info(f"Fitting PySpark median Imputer across {len(available_numeric)} numeric columns...")
        imputer = Imputer(
            inputCols=available_numeric,
            outputCols=available_numeric,
        ).setStrategy("median")

        self.imputer_model = imputer.fit(df)

        # Extract and persist surrogate medians for zero-leakage verification
        surrogate_df = self.imputer_model.surrogateDF
        fitted = {}
        for c in available_numeric:
            if c in surrogate_df.columns:
                val = surrogate_df.select(c).first()
                if val is not None and val[0] is not None:
                    fitted[c] = float(val[0])
        self.fitted_medians = fitted
        logger.info(f"Successfully learned training medians for {len(self.fitted_medians)} features.")

    def apply_imputer(self, df: DataFrame) -> DataFrame:
        """Apply fitted median imputer model to DataFrame with zero data leakage."""
        if self.imputer_model is not None:
            df = self.imputer_model.transform(df)

        # Ensure all continuous and indicator columns are cast to double with no remaining nulls
        for col_name in self.numeric_cols:
            if col_name in df.columns:
                fallback_val = self.fitted_medians.get(col_name, 0.0)
                df = df.withColumn(
                    col_name,
                    F.coalesce(F.col(col_name).cast("double"), F.lit(fallback_val)),
                )

        for col_name in self.binary_indicator_cols:
            if col_name in df.columns:
                df = df.withColumn(
                    col_name,
                    F.coalesce(F.col(col_name).cast("double"), F.lit(0.0)),
                )

        return df

    def fit_outlier_bounds(
        self,
        df: DataFrame,
        columns: Optional[List[str]] = None,
        iqr_multiplier: float = 3.0,
    ) -> None:
        """Calculate IQR (25%, 75%) outlier bounds on the training split only."""
        if columns is None:
            # Outlier treatment applied to key continuous network rate and volume metrics
            columns = [
                c for c in self.numeric_cols
                if c in df.columns and c not in ("id", "label", "sttl", "dttl", "swin", "dwin")
            ]

        logger.info(f"Computing zero-leakage IQR outlier bounds for {len(columns)} numeric columns...")
        bounds: Dict[str, Tuple[float, float]] = {}

        for col_name in columns:
            try:
                quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
                if len(quantiles) == 2:
                    q25, q75 = quantiles[0], quantiles[1]
                    iqr = q75 - q25
                    lower = max(0.0, q25 - (iqr_multiplier * iqr))
                    upper = q75 + (iqr_multiplier * iqr)
                    # Guard against zero-IQR distributions
                    if upper <= lower:
                        q99 = df.approxQuantile(col_name, [0.999], 0.01)[0]
                        upper = max(q99, q75, 1.0)
                    bounds[col_name] = (float(lower), float(upper))
            except Exception as e:
                logger.warning(f"Could not compute quantiles for {col_name}: {e}")

        self.outlier_bounds = bounds
        logger.info(f"Fitted outlier clipping boundaries for {len(self.outlier_bounds)} columns.")

    def apply_outlier_capping(self, df: DataFrame) -> DataFrame:
        """Apply soft-capping/clipping to numeric features without dropping attack instances."""
        capped_df = df
        for col_name, (lower, upper) in self.outlier_bounds.items():
            if col_name in capped_df.columns:
                capped_df = capped_df.withColumn(
                    col_name,
                    F.when(F.col(col_name) > upper, F.lit(upper))
                    .when(F.col(col_name) < lower, F.lit(lower))
                    .otherwise(F.col(col_name)),
                )
        return capped_df

    def split(
        self,
        df: DataFrame,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Split dataset into Train, Validation, and Test partitions with reproducible seed."""
        t = train_ratio or self.splits_cfg.get("train_ratio", 0.70)
        v = val_ratio or self.splits_cfg.get("validation_ratio", 0.15)
        te = test_ratio or self.splits_cfg.get("test_ratio", 0.15)
        s = seed or self.splits_cfg.get("seed", 42)

        logger.info(f"Splitting dataset: Train={t:.0%}, Val={v:.0%}, Test={te:.0%} (Seed={s})")
        train_df, val_df, test_df = df.randomSplit([t, v, te], seed=s)
        return train_df, val_df, test_df

    def clean(self, df: DataFrame) -> DataFrame:
        """Basic sanitation pipeline: deduplication, string standardization, and null fill."""
        logger.info("Executing basic preprocessing clean...")
        df = self.remove_duplicates(df)
        df = self.handle_categoricals(df)

        # Fallback numeric null imputation
        for col_name in self.numeric_cols:
            if col_name in df.columns:
                df = df.withColumn(
                    col_name,
                    F.coalesce(F.col(col_name).cast("double"), F.lit(0.0)),
                )
        return df

    def fit_transform_train(self, train_df: DataFrame) -> DataFrame:
        """Fit imputer and outlier parameters solely on training data, then transform."""
        logger.info("Executing fit_transform on training partition...")
        cleaned_train = self.handle_categoricals(train_df)
        self.fit_imputer(cleaned_train)
        imputed_train = self.apply_imputer(cleaned_train)
        self.fit_outlier_bounds(imputed_train)
        capped_train = self.apply_outlier_capping(imputed_train)
        return capped_train

    def transform_eval(self, eval_df: DataFrame) -> DataFrame:
        """Transform validation or holdout test partitions using strictly training-fitted parameters."""
        cleaned_eval = self.handle_categoricals(eval_df)
        imputed_eval = self.apply_imputer(cleaned_eval)
        capped_eval = self.apply_outlier_capping(imputed_eval)
        return capped_eval

    def get_fitted_parameters(self) -> Dict[str, Any]:
        """Return fitted imputation medians and outlier bounds for zero-leakage auditing."""
        return {
            "duplicate_records_removed": self.duplicate_count,
            "fitted_medians": self.fitted_medians,
            "outlier_bounds": {
                k: {"lower": v[0], "upper": v[1]} for k, v in self.outlier_bounds.items()
            },
        }

    def save_fitted_parameters(self, output_path: str) -> None:
        """Serialize fitted parameters to disk for audit trails and serving inference."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.get_fitted_parameters(), f, indent=2)
        logger.info(f"Saved zero-leakage preprocessing parameters to: {output_path}")
