"""
==============================================================================
SentinelSpark — Network Feature Engineering & Target Leakage Protection
==============================================================================
Computes domain-specific network intrusion features: packet symmetry,
byte transmission rates, TCP handshake dynamics, and safe-division volume metrics.
Guarantees strict target leakage protection across Stage 1 and Stage 2 models.
==============================================================================
"""

from typing import Any, Dict, List
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.utils import logger


class FeatureEngineer:
    """Network-domain behavioral feature engineering with strict leakage boundaries."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feat_cfg = config.get("features", {})
        self.continuous_cols = self.feat_cfg.get("continuous_numeric_cols", [])
        self.categorical_cols = self.feat_cfg.get("categorical_cols", ["proto", "service", "state"])
        self.binary_indicator_cols = self.feat_cfg.get("binary_indicator_cols", ["is_ftp_login", "is_sm_ips_ports"])

    def get_engineered_feature_names(self) -> List[str]:
        """Return canonical list of engineered interaction and safe-division features."""
        return [
            # Directional packet / byte asymmetries
            "bytes_per_packet_src",
            "bytes_per_packet_dst",
            "packet_ratio",
            "byte_ratio",
            "tcp_handshake_ratio",
            "total_packets",
            "total_bytes",
            # Rate and safe-division volume dynamics
            "packet_rate",
            "byte_rate",
            "avg_bytes_per_packet",
            "sbyte_ratio",
            # Advanced Domain-Specific Ratios
            "packet_size_ratio",
            "load_ratio",
            "ttl_ratio",
            "tcprtt_ratio",
        ]

    def drop_non_predictive_columns(self, df: DataFrame) -> DataFrame:
        """Drop non-predictive identifiers and row indices to eliminate data leakage."""
        non_predictive = ["id", "index", "_c0", "_c1"]
        cols_to_drop = [c for c in non_predictive if c in df.columns]
        if cols_to_drop:
            logger.info(f"Target leakage protection: Dropping non-predictive columns: {cols_to_drop}")
            df = df.drop(*cols_to_drop)
        return df

    def filter_stage2_threats(self, df: DataFrame) -> DataFrame:
        """Filter dataset strictly for Stage 2 threat classification.

        Excludes legitimate traffic (label == 0 or attack_cat == 'Normal') to prevent
        multiclass model contamination, ensuring Stage 2 classifies only genuine incursions.
        """
        logger.info("Stage 2 Target Isolation: Filtering out legitimate 'Normal' records...")
        filtered_df = df.filter(
            (F.col("label") == 1)
            & (F.col("attack_cat") != "Normal")
            & (F.col("attack_cat").isNotNull())
        )
        return filtered_df

    def engineer_features(self, df: DataFrame) -> DataFrame:
        """Compute safe-division network behavioral features and rates."""
        logger.info("Computing network behavioral features (rates, ratios, asymmetries)...")
        eps = 1e-6

        engineered_df = (
            df
            # 1. Source and Destination Bytes Per Packet
            .withColumn(
                "bytes_per_packet_src",
                F.col("sbytes") / (F.col("spkts") + F.lit(eps)),
            )
            .withColumn(
                "bytes_per_packet_dst",
                F.col("dbytes") / (F.col("dpkts") + F.lit(eps)),
            )
            # 2. Packet and Byte Balance Ratios
            .withColumn(
                "packet_ratio",
                (F.col("spkts") + F.lit(1.0)) / (F.col("dpkts") + F.lit(1.0)),
            )
            .withColumn(
                "byte_ratio",
                (F.col("sbytes") + F.lit(1.0)) / (F.col("dbytes") + F.lit(1.0)),
            )
            # 3. TCP Handshake Round-Trip Timing Ratio
            .withColumn(
                "tcp_handshake_ratio",
                (F.col("synack") + F.lit(eps)) / (F.col("tcprtt") + F.lit(eps)),
            )
            # 4. Aggregated Flow Volume
            .withColumn("total_packets", F.col("spkts") + F.col("dpkts"))
            .withColumn("total_bytes", F.col("sbytes") + F.col("dbytes"))
            # 5. Safe-Division Rate Features (Handling 0 duration safely)
            .withColumn(
                "packet_rate",
                (F.col("spkts") + F.col("dpkts")) / (F.col("dur") + F.lit(eps)),
            )
            .withColumn(
                "byte_rate",
                (F.col("sbytes") + F.col("dbytes")) / (F.col("dur") + F.lit(eps)),
            )
            .withColumn(
                "avg_bytes_per_packet",
                (F.col("sbytes") + F.col("dbytes"))
                / (F.col("spkts") + F.col("dpkts") + F.lit(eps)),
            )
            .withColumn(
                "sbyte_ratio",
                F.col("sbytes") / (F.col("sbytes") + F.col("dbytes") + F.lit(eps)),
            )
            # 6. Advanced Domain Ratios (packet size, load, ttl, tcprtt)
            .withColumn(
                "packet_size_ratio",
                F.col("smean") / (F.col("dmean") + F.lit(eps)),
            )
            .withColumn(
                "load_ratio",
                F.col("sload") / (F.col("dload") + F.lit(eps)),
            )
            .withColumn(
                "ttl_ratio",
                F.col("sttl") / (F.col("dttl") + F.lit(eps)),
            )
            .withColumn(
                "tcprtt_ratio",
                F.col("synack") / (F.col("ackdat") + F.lit(eps)),
            )
        )

        return engineered_df

    def transform(self, df: DataFrame) -> DataFrame:
        """Pipeline-compliant transformation alias."""
        return self.engineer_features(df)
