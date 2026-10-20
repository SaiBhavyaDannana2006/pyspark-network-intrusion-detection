"""
==============================================================================
SentinelSpark — Distributed Exploratory Data Analysis (EDA)
==============================================================================
High-performance distributed dataset profiling using native Apache Spark SQL
and DataFrame transformations for UNSW-NB15 network intrusion detection.

Features:
- Schema, dimension, and partition distribution inspection.
- Null, NaN, empty string, and placeholder profiling in a single distributed pass.
- Binary target (0 vs 1) and multiclass attack category frequency & ratio analysis.
- Distributed numeric summary statistics (mean, stddev, min, quantiles, max) via Spark summary().
- Categorical distribution metrics for protocol, service, and connection states.
- Seamless CSV export and structured Python dict outputs for CLI and reporting.
==============================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.utils import logger


class DataExplorer:
    """Distributed EDA tools using native Apache Spark SQL and DataFrame operations."""

    def __init__(self, df: DataFrame, config: Optional[Dict[str, Any]] = None):
        self.df = df
        self.config = config or {}

    def get_schema_overview(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Inspect schema details, record counts, and partition topology using PySpark."""
        total_records = self.df.count()
        columns = self.df.columns
        num_partitions = self.df.rdd.getNumPartitions()

        schema_records = []
        for field in self.df.schema.fields:
            schema_records.append({
                "column_name": field.name,
                "data_type": str(field.dataType),
                "nullable": field.nullable,
            })
        schema_df = pd.DataFrame(schema_records)

        overview: Dict[str, Any] = {
            "total_records": total_records,
            "total_columns": len(columns),
            "num_partitions": num_partitions,
            "columns": columns,
        }

        logger.info(
            f"Schema Overview: {total_records:,} records, {len(columns)} columns across {num_partitions} partitions."
        )
        return overview, schema_df

    def get_basic_stats(self) -> Dict[str, Any]:
        """Backward-compatible helper returning dataset dimensions and partition metrics."""
        overview, _ = self.get_schema_overview()
        return {
            "total_records": overview["total_records"],
            "total_features": overview["total_columns"],
            "num_partitions": overview["num_partitions"],
            "columns": overview["columns"],
        }

    def audit_missing_and_empty(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Profile null, NaN, empty string, and '-' placeholder frequencies across all columns.

        Uses a single distributed aggregation query across the entire dataset to maximize
        throughput without multiple cluster shuffles.
        """
        total_records = self.df.count()
        agg_exprs = []

        for col_name, dtype in self.df.dtypes:
            # 1. Null / NaN condition
            if dtype in ("float", "double"):
                null_cond = F.col(col_name).isNull() | F.isnan(F.col(col_name))
            else:
                null_cond = F.col(col_name).isNull()
            agg_exprs.append(F.sum(F.when(null_cond, 1).otherwise(0)).alias(f"{col_name}__null"))

            # 2. String-specific empty / '-' placeholder condition
            if dtype == "string":
                empty_cond = F.trim(F.col(col_name)) == ""
                dash_cond = F.trim(F.col(col_name)) == "-"
                agg_exprs.append(F.sum(F.when(empty_cond, 1).otherwise(0)).alias(f"{col_name}__empty"))
                agg_exprs.append(F.sum(F.when(dash_cond, 1).otherwise(0)).alias(f"{col_name}__dash"))

        # Single distributed pass
        agg_row = self.df.agg(*agg_exprs).collect()[0].asDict()

        audit_records = []
        for col_name, dtype in self.df.dtypes:
            n_null = int(agg_row.get(f"{col_name}__null", 0) or 0)
            n_empty = int(agg_row.get(f"{col_name}__empty", 0) or 0)
            n_dash = int(agg_row.get(f"{col_name}__dash", 0) or 0)

            audit_records.append({
                "column_name": col_name,
                "data_type": dtype,
                "null_count": n_null,
                "null_pct": round((n_null / total_records * 100) if total_records > 0 else 0.0, 2),
                "empty_string_count": n_empty,
                "empty_string_pct": round((n_empty / total_records * 100) if total_records > 0 else 0.0, 2),
                "dash_placeholder_count": n_dash,
                "dash_placeholder_pct": round((n_dash / total_records * 100) if total_records > 0 else 0.0, 2),
            })

        audit_df = pd.DataFrame(audit_records)
        audit_dict: Dict[str, Any] = {
            "total_records": total_records,
            "columns_with_nulls": [r for r in audit_records if r["null_count"] > 0],
            "columns_with_empty_strings": [r for r in audit_records if r["empty_string_count"] > 0],
            "columns_with_dash_placeholders": [r for r in audit_records if r["dash_placeholder_count"] > 0],
            "column_metrics": {r["column_name"]: r for r in audit_records},
        }

        return audit_dict, audit_df

    def check_missing_values(self) -> List[Dict[str, Any]]:
        """Backward-compatible helper returning list of columns with null counts."""
        audit_dict, _ = self.audit_missing_and_empty()
        return [
            {
                "column": r["column_name"],
                "null_count": r["null_count"],
                "null_pct": r["null_pct"],
            }
            for r in audit_dict.get("columns_with_nulls", [])
        ]

    def analyze_target_distribution(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Compute binary label (0: Normal, 1: Attack) distribution and ratios using PySpark."""
        total = self.df.count()
        target_spark = (
            self.df.groupBy("label")
            .agg(F.count("*").alias("count"))
            .withColumn("ratio", F.round(F.col("count") / F.lit(total), 4))
            .withColumn("percentage", F.round(F.col("ratio") * 100.0, 2))
            .orderBy("label")
        )
        target_df = target_spark.toPandas()

        target_dict: Dict[str, Any] = {
            "total_records": total,
            "distribution": target_df.to_dict(orient="records"),
            "normal_count": int(target_df.loc[target_df["label"] == 0, "count"].values[0]) if (target_df["label"] == 0).any() else 0,
            "attack_count": int(target_df.loc[target_df["label"] == 1, "count"].values[0]) if (target_df["label"] == 1).any() else 0,
        }
        return target_dict, target_df

    def analyze_multiclass_distribution(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Compute distribution of multiclass attack categories using PySpark."""
        total = self.df.count()
        multi_spark = (
            self.df.groupBy("attack_cat")
            .agg(F.count("*").alias("count"))
            .withColumn("ratio", F.round(F.col("count") / F.lit(total), 4))
            .withColumn("percentage", F.round(F.col("ratio") * 100.0, 2))
            .orderBy(F.desc("count"))
        )
        multi_df = multi_spark.toPandas()

        multi_dict: Dict[str, Any] = {
            "total_records": total,
            "num_categories": len(multi_df),
            "attack_categories": multi_df.to_dict(orient="records"),
        }
        return multi_dict, multi_df

    def get_class_distributions(self) -> Dict[str, Any]:
        """Backward-compatible helper returning binary and multiclass distributions."""
        _, binary_df = self.analyze_target_distribution()
        _, multi_df = self.analyze_multiclass_distribution()
        return {
            "binary_distribution": binary_df.to_dict(orient="records"),
            "attack_distribution": multi_df.to_dict(orient="records"),
        }

    def compute_numeric_summary(
        self, columns: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Calculate distributed summary statistics (count, mean, stddev, min, quantiles, max) via Spark summary()."""
        if columns is None:
            numeric_types = ("int", "bigint", "smallint", "tinyint", "float", "double", "decimal", "long")
            columns = [c for c, t in self.df.dtypes if any(t.lower().startswith(nt) for nt in numeric_types)]
            # Exclude target and id columns from default summary list
            columns = [c for c in columns if c not in ("id", "label")]

        logger.info(f"Computing distributed summary statistics for {len(columns)} numeric columns...")
        summary_spark = self.df.select(*columns).summary()
        summary_df = summary_spark.toPandas()
        summary_dict = summary_df.set_index("summary").to_dict()

        return summary_dict, summary_df

    def analyze_categorical_distributions(
        self, top_n: int = 10, columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Compute frequency distributions for protocol, service, and connection state categories."""
        if columns is None:
            columns = ["proto", "service", "state"]

        total = self.df.count()
        cat_dfs: Dict[str, pd.DataFrame] = {}

        for col_name in columns:
            if col_name in self.df.columns:
                cat_spark = (
                    self.df.groupBy(col_name)
                    .agg(F.count("*").alias("count"))
                    .withColumn("percentage", F.round((F.col("count") / F.lit(total)) * 100.0, 2))
                    .orderBy(F.desc("count"))
                    .limit(top_n)
                )
                cat_dfs[col_name] = cat_spark.toPandas()

        return cat_dfs

    def run_full_exploration(self, export_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute complete distributed EDA workflow and export profiling artifacts to CSV if requested."""
        logger.info("Initiating full distributed EDA workflow...")

        overview_dict, schema_df = self.get_schema_overview()
        missing_dict, missing_df = self.audit_missing_and_empty()
        target_dict, target_df = self.analyze_target_distribution()
        multi_dict, multi_df = self.analyze_multiclass_distribution()
        summary_dict, summary_df = self.compute_numeric_summary()
        cat_dfs = self.analyze_categorical_distributions()

        if export_dir:
            out_path = Path(export_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Exporting EDA summaries to CSV in: {out_path}")

            schema_df.to_csv(out_path / "schema_overview.csv", index=False)
            missing_df.to_csv(out_path / "missing_values_audit.csv", index=False)
            target_df.to_csv(out_path / "binary_target_distribution.csv", index=False)
            multi_df.to_csv(out_path / "multiclass_category_distribution.csv", index=False)
            summary_df.to_csv(out_path / "numeric_summary_statistics.csv", index=False)

            for col_name, cat_df in cat_dfs.items():
                cat_df.to_csv(out_path / f"{col_name}_distribution.csv", index=False)

        return {
            "overview": overview_dict,
            "missing_audit": missing_dict,
            "binary_target": target_dict,
            "multiclass_target": multi_dict,
            "numeric_summary": summary_dict,
            "categorical_distributions": {k: v.to_dict(orient="records") for k, v in cat_dfs.items()},
        }
