"""
==============================================================================
SentinelSpark — Granular Attack Signature & Behavioral Vector Analysis
==============================================================================
Performs distributed traffic analysis across protocols, services, and attack
classes; investigates top confused attack vectors and numeric flow statistics.
==============================================================================
"""

from typing import Any, Dict, Optional
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.utils import logger


class AttackAnalyzer:
    """Detailed threat vector, service vulnerability, and confusion analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, df: Optional[DataFrame] = None):
        self.config = config or {}
        self.df = df

    def _resolve_df(self, df: Optional[DataFrame]) -> DataFrame:
        """Resolve DataFrame from argument or instance property."""
        resolved = df if df is not None else self.df
        if resolved is None:
            raise ValueError("No DataFrame provided to AttackAnalyzer.")
        return resolved

    def analyze_protocol_attack_rates(
        self, df: Optional[DataFrame] = None, top_n: int = 10
    ) -> pd.DataFrame:
        """Calculate intrusion attack rate and volume across transport protocols."""
        target_df = self._resolve_df(df)
        logger.info(f"Computing attack rates for top {top_n} network protocols...")

        proto_stats = (
            target_df.groupBy("proto")
            .agg(
                F.count("*").alias("total_flows"),
                F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).alias("attack_flows"),
            )
            .withColumn(
                "attack_rate",
                F.round((F.col("attack_flows") / F.col("total_flows")) * 100.0, 2),
            )
            .orderBy(F.desc("total_flows"))
            .limit(top_n)
        )
        return proto_stats.toPandas()

    def analyze_service_attack_rates(
        self, df: Optional[DataFrame] = None, top_n: int = 10
    ) -> pd.DataFrame:
        """Calculate intrusion attack density across network application services."""
        target_df = self._resolve_df(df)
        logger.info(f"Computing attack rates for top {top_n} application services...")

        service_stats = (
            target_df.groupBy("service")
            .agg(
                F.count("*").alias("total_flows"),
                F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).alias("attack_flows"),
            )
            .withColumn(
                "attack_rate",
                F.round((F.col("attack_flows") / F.col("total_flows")) * 100.0, 2),
            )
            .orderBy(F.desc("total_flows"))
            .limit(top_n)
        )
        return service_stats.toPandas()

    def compute_traffic_stats_per_class(
        self, df: Optional[DataFrame] = None
    ) -> pd.DataFrame:
        """Compute average flow duration, packet volumes, and byte rates per attack taxonomy class."""
        target_df = self._resolve_df(df)
        logger.info("Computing network flow behavioral profiles per attack category...")

        # Build list of aggregations based on present columns
        aggs = [F.count("*").alias("flow_count")]
        if "dur" in target_df.columns:
            aggs.append(F.round(F.avg("dur"), 4).alias("avg_duration"))
        if "sbytes" in target_df.columns:
            aggs.append(F.round(F.avg("sbytes"), 1).alias("avg_sbytes"))
        if "dbytes" in target_df.columns:
            aggs.append(F.round(F.avg("dbytes"), 1).alias("avg_dbytes"))
        if "rate" in target_df.columns:
            aggs.append(F.round(F.avg("rate"), 2).alias("avg_rate"))
        if "spkts" in target_df.columns:
            aggs.append(F.round(F.avg("spkts"), 1).alias("avg_spkts"))
        if "dpkts" in target_df.columns:
            aggs.append(F.round(F.avg("dpkts"), 1).alias("avg_dpkts"))

        class_stats = (
            target_df.groupBy("attack_cat")
            .agg(*aggs)
            .orderBy(F.desc("flow_count"))
        )
        return class_stats.toPandas()

    def find_top_confused_categories(
        self, predictions_df: DataFrame, top_n: int = 10
    ) -> pd.DataFrame:
        """Identify and rank the most frequently misclassified attack categories."""
        logger.info("Identifying top confused attack categories in predictions...")
        actual_col = "attack_cat" if "attack_cat" in predictions_df.columns else "label"
        pred_col = "predicted_attack_cat" if "predicted_attack_cat" in predictions_df.columns else "prediction"

        confused_df = (
            predictions_df.filter(F.col(actual_col) != F.col(pred_col))
            .groupBy(actual_col, pred_col)
            .agg(F.count("*").alias("confusion_count"))
            .withColumnRenamed(actual_col, "actual_category")
            .withColumnRenamed(pred_col, "predicted_category")
            .orderBy(F.desc("confusion_count"))
            .limit(top_n)
        )
        return confused_df.toPandas()

    # Legacy support
    def analyze_attacks_by_protocol(self) -> DataFrame:
        """Legacy helper returning PySpark DataFrame."""
        return (
            self._resolve_df(None).filter(F.col("label") == 1)
            .groupBy("proto", "attack_cat")
            .agg(F.count("*").alias("attack_count"))
            .orderBy(F.desc("attack_count"))
        )

    def analyze_service_vulnerabilities(self) -> DataFrame:
        """Legacy helper returning PySpark DataFrame."""
        return (
            self._resolve_df(None).groupBy("service")
            .agg(
                F.count("*").alias("total_flows"),
                F.sum("label").alias("malicious_flows"),
                F.round((F.sum("label") / F.count("*")) * 100, 2).alias("malicious_percentage"),
            )
            .orderBy(F.desc("total_flows"))
        )
