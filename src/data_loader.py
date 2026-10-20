"""
==============================================================================
SentinelSpark — Distributed Data Ingestion & Synthetic Fallback Engine
==============================================================================
Handles discovery, schema enforcement, distributed loading, and synthetic
generation of UNSW-NB15 network intrusion datasets.
==============================================================================
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.utils import logger, PROJECT_ROOT


def get_unsw_nb15_schema() -> StructType:
    """Return explicit canonical StructType schema for the 45-column UNSW-NB15 dataset."""
    return StructType(
        [
            StructField("id", LongType(), True),
            StructField("dur", DoubleType(), True),
            StructField("proto", StringType(), True),
            StructField("service", StringType(), True),
            StructField("state", StringType(), True),
            StructField("spkts", LongType(), True),
            StructField("dpkts", LongType(), True),
            StructField("sbytes", LongType(), True),
            StructField("dbytes", LongType(), True),
            StructField("rate", DoubleType(), True),
            StructField("sttl", IntegerType(), True),
            StructField("dttl", IntegerType(), True),
            StructField("sload", DoubleType(), True),
            StructField("dload", DoubleType(), True),
            StructField("sloss", LongType(), True),
            StructField("dloss", LongType(), True),
            StructField("sinpkt", DoubleType(), True),
            StructField("dinpkt", DoubleType(), True),
            StructField("sjit", DoubleType(), True),
            StructField("djit", DoubleType(), True),
            StructField("swin", LongType(), True),
            StructField("stcpb", LongType(), True),
            StructField("dtcpb", LongType(), True),
            StructField("dwin", LongType(), True),
            StructField("tcprtt", DoubleType(), True),
            StructField("synack", DoubleType(), True),
            StructField("ackdat", DoubleType(), True),
            StructField("smean", DoubleType(), True),
            StructField("dmean", DoubleType(), True),
            StructField("trans_depth", LongType(), True),
            StructField("response_body_len", LongType(), True),
            StructField("ct_srv_src", IntegerType(), True),
            StructField("ct_state_ttl", IntegerType(), True),
            StructField("ct_dst_ltm", IntegerType(), True),
            StructField("ct_src_dport_ltm", IntegerType(), True),
            StructField("ct_dst_sport_ltm", IntegerType(), True),
            StructField("ct_dst_src_ltm", IntegerType(), True),
            StructField("is_ftp_login", IntegerType(), True),
            StructField("ct_ftp_cmd", IntegerType(), True),
            StructField("ct_flw_http_mthd", IntegerType(), True),
            StructField("ct_src_ltm", IntegerType(), True),
            StructField("ct_srv_dst", IntegerType(), True),
            StructField("is_sm_ips_ports", IntegerType(), True),
            StructField("attack_cat", StringType(), True),
            StructField("label", IntegerType(), True),
        ]
    )


class DataLoader:
    """Manages distributed file discovery, schema normalization, and ingestion for UNSW-NB15."""

    def __init__(self, spark: SparkSession, config: Optional[Dict[str, Any]] = None):
        self.spark = spark
        self.config = config or {}
        self.data_cfg = self.config.get("data", {})
        raw_dir_val = self.data_cfg.get("raw_dir", "data/raw")
        self.raw_dir = PROJECT_ROOT / raw_dir_val if not Path(raw_dir_val).is_absolute() else Path(raw_dir_val)
        self.metadata: Dict[str, Any] = {
            "is_synthetic": False,
            "benchmark_valid": False,
            "source": "Uninitialized",
        }

    def discover_dataset(
        self, custom_path: Optional[str] = None
    ) -> Optional[Union[str, List[str], Tuple[str, str]]]:
        """Discover UNSW-NB15 dataset files from CLI override, env var, or raw directory.

        Discovery precedence:
        1. Explicit custom_path argument.
        2. Environment variable DATASET_PATH.
        3. Standard train/test splits (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv).
        4. Raw partitions (UNSW-NB15_1.csv through UNSW-NB15_4.csv).
        5. Combined dataset (UNSW-NB15_combined.csv).
        6. Any CSV files inside the raw_dir directory.
        """
        # 1. Direct custom path override
        if custom_path:
            p = Path(custom_path)
            if p.exists():
                logger.info(f"Using explicitly specified dataset path: {custom_path}")
                return str(p)
            logger.warning(f"Specified dataset path does not exist: {custom_path}")

        # 2. Environment variable override
        env_path = os.environ.get("DATASET_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists():
                logger.info(f"Discovered dataset path from DATASET_PATH env var: {env_path}")
                return str(p)
            logger.warning(f"DATASET_PATH environment variable path does not exist: {env_path}")

        if not self.raw_dir.exists():
            return None

        # 3. Check for standard training/testing split files
        train_name = self.data_cfg.get("discovery_files", {}).get(
            "train_filename", "UNSW_NB15_training-set.csv"
        )
        test_name = self.data_cfg.get("discovery_files", {}).get(
            "test_filename", "UNSW_NB15_testing-set.csv"
        )
        train_file = self.raw_dir / train_name
        test_file = self.raw_dir / test_name

        if train_file.exists() and test_file.exists():
            logger.info(f"Discovered official UNSW-NB15 train/test splits: {train_file.name}, {test_file.name}")
            return str(train_file), str(test_file)

        # 4. Check for raw partitions (UNSW-NB15_1.csv through 4.csv or UNSW_NB15_1..4.csv)
        partition_patterns = ["UNSW-NB15_[1-4].csv", "UNSW_NB15_[1-4].csv"]
        partitions: List[str] = []
        for pat in partition_patterns:
            matched = sorted(str(p) for p in self.raw_dir.glob(pat))
            if matched:
                partitions = matched
                break

        if partitions:
            logger.info(f"Discovered {len(partitions)} raw UNSW-NB15 partitions: {[Path(p).name for p in partitions]}")
            return partitions

        # 5. Check for combined dataset file
        combined_name = self.data_cfg.get("discovery_files", {}).get(
            "combined_filename", "UNSW-NB15_combined.csv"
        )
        combined_file = self.raw_dir / combined_name
        if combined_file.exists():
            logger.info(f"Discovered combined UNSW-NB15 dataset: {combined_file.name}")
            return str(combined_file)

        # 6. Fallback to any CSV in raw_dir
        csv_files = list(self.raw_dir.glob("*.csv"))
        if csv_files:
            logger.info(f"Discovered CSV file in raw directory: {csv_files[0].name}")
            return str(csv_files[0])

        return None

    def normalize_schema(self, df: DataFrame) -> DataFrame:
        """Inspect and normalize DataFrame headers and column types to canonical 45-column schema.

        - Strips whitespace and normalizes column headers to lowercase.
        - Detects missing/generic headers (_c0, _c1, ...) and maps to UNSW-NB15 schema.
        - Casts existing columns to canonical PySpark data types.
        - Adds default/monotonic columns (such as 'id') if absent.
        """
        canonical_schema = get_unsw_nb15_schema()
        canonical_fields = {f.name: f for f in canonical_schema.fields}
        canonical_names = [f.name for f in canonical_schema.fields]

        # 1. Clean and lowercase column headers
        current_cols = df.columns
        cleaned_cols = [c.strip().lower().replace('"', "").replace("'", "") for c in current_cols]
        df = df.toDF(*cleaned_cols)

        # 2. Check if headers are missing/generic (_c0, _c1, ...)
        is_generic_headers = all(c.startswith("_c") for c in df.columns[: min(5, len(df.columns))])
        if is_generic_headers:
            logger.info("Detected generic column headers without header row; mapping to UNSW-NB15 schema.")
            if len(df.columns) == len(canonical_names):
                df = df.toDF(*canonical_names)
            elif len(df.columns) == 49:
                # Raw UNSW-NB15 with 49 columns (includes srcip, sport, dstip, dsport)
                raw_49_cols = [
                    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
                    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
                    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb",
                    "dtcpb", "smean", "dmean", "trans_depth", "response_body_len",
                    "sjit", "djit", "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat",
                    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
                    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat",
                    "label"
                ]
                if len(df.columns) == len(raw_49_cols):
                    df = df.toDF(*raw_49_cols)

        # 3. Ensure 'id' column exists
        if "id" not in df.columns:
            df = df.withColumn("id", F.monotonically_increasing_id().cast(LongType()))

        # 4. Cast and align existing columns to canonical schema
        select_exprs = []
        for name in canonical_names:
            if name in df.columns:
                target_type = canonical_fields[name].dataType
                select_exprs.append(F.col(name).cast(target_type).alias(name))
            else:
                # Provide safe fallback column with null/defaults if missing
                target_field = canonical_fields[name]
                logger.warning(f"Column '{name}' missing from input data; filling with nulls.")
                select_exprs.append(F.lit(None).cast(target_field.dataType).alias(name))

        return df.select(*select_exprs)

    def load_or_generate(self, custom_path: Optional[str] = None) -> DataFrame:
        """Load discovered dataset into Spark DataFrame or generate synthetic fallback."""
        discovered = self.discover_dataset(custom_path=custom_path)

        if discovered is not None:
            if isinstance(discovered, tuple):
                # Standard train/test split pair
                train_p, test_p = discovered
                logger.info(f"Loading official train split: {train_p}")
                train_df = self.spark.read.option("header", "true").csv(train_p)
                logger.info(f"Loading official test split: {test_p}")
                test_df = self.spark.read.option("header", "true").csv(test_p)
                raw_df = train_df.unionByName(test_df, allowMissingColumns=True)
                source_desc = f"Split: {Path(train_p).name} + {Path(test_p).name}"
            elif isinstance(discovered, list):
                # Multi-partition files (UNSW-NB15_1..4)
                logger.info(f"Loading {len(discovered)} partition files...")
                raw_df = self.spark.read.option("header", "true").csv(discovered)
                source_desc = f"Partitions: {len(discovered)} files"
            else:
                # Single CSV file
                logger.info(f"Loading single dataset file: {discovered}")
                raw_df = self.spark.read.option("header", "true").csv(str(discovered))
                source_desc = str(discovered)

            normalized_df = self.normalize_schema(raw_df)
            self.metadata = {
                "is_synthetic": False,
                "benchmark_valid": True,
                "source": source_desc,
                "loaded_at": datetime.now().isoformat(),
                "schema_columns": len(normalized_df.columns),
            }
            logger.info(f"Successfully loaded and normalized dataset from: {source_desc}")
            return normalized_df

        # If files not detected, check synthetic fallback
        if self.data_cfg.get("allow_synthetic_fallback", True):
            sample_size = self.data_cfg.get("synthetic_sample_size", 10000)
            logger.warning(
                "UNSW-NB15 raw CSV files not detected. Generating synthetic benchmark dataset "
                f"({sample_size:,} records)..."
            )
            df, meta = self.generate_synthetic_data(num_samples=sample_size)
            self.metadata = meta
            return df

        raise FileNotFoundError(
            f"No UNSW-NB15 dataset found in '{self.raw_dir}' and synthetic fallback is disabled."
        )

    def load_csv(self, file_path: Union[str, Path]) -> DataFrame:
        """Load a single CSV file and normalize to canonical UNSW-NB15 schema."""
        raw_df = self.spark.read.option("header", "true").csv(str(file_path))
        return self.normalize_schema(raw_df)

    def generate_synthetic_data(
        self, num_samples: int = 10000, seed: int = 42
    ) -> Tuple[DataFrame, Dict[str, Any]]:
        """Generate statistically realistic synthetic UNSW-NB15 network flow records.

        Explicitly flags synthetic datasets in metadata so they are never represented
        as official benchmark results.
        """
        logger.info(f"Synthesizing {num_samples:,} distributed UNSW-NB15 flow records (Seed: {seed})...")

        base_df = self.spark.range(0, num_samples).withColumnRenamed("id", "id")

        protocols = ["tcp", "udp", "arp", "ospf", "icmp", "sctp"]
        services = ["-", "http", "ftp", "smtp", "dns", "ssh", "ssl", "dhcp", "pop3"]
        states = ["FIN", "CON", "INT", "REQ", "RST", "ACC", "CLO"]
        attacks = [
            "Normal",
            "Generic",
            "Exploits",
            "Fuzzers",
            "DoS",
            "Reconnaissance",
            "Analysis",
            "Backdoor",
            "Shellcode",
            "Worms",
        ]

        # Use deterministic hash expressions for realistic reproducible network patterns
        indexed_df = (
            base_df
            .withColumn("proto_idx", (F.abs(F.hash(F.col("id") + F.lit(11 + seed))) % len(protocols)))
            .withColumn("service_idx", (F.abs(F.hash(F.col("id") + F.lit(23 + seed))) % len(services)))
            .withColumn("state_idx", (F.abs(F.hash(F.col("id") + F.lit(37 + seed))) % len(states)))
            .withColumn("attack_idx", (F.abs(F.hash(F.col("id") + F.lit(97 + seed))) % len(attacks)))
            .withColumn(
                "label",
                F.when(F.col("attack_idx") == 0, F.lit(0)).otherwise(F.lit(1)),
            )
        )

        # Map indices to categorical strings
        proto_expr = F.element_at(F.array(*[F.lit(p) for p in protocols]), F.col("proto_idx") + 1)
        service_expr = F.element_at(F.array(*[F.lit(s) for s in services]), F.col("service_idx") + 1)
        state_expr = F.element_at(F.array(*[F.lit(st) for st in states]), F.col("state_idx") + 1)
        attack_expr = F.element_at(F.array(*[F.lit(a) for a in attacks]), F.col("attack_idx") + 1)

        # Assemble full 45-column UNSW-NB15 schema
        synthetic_df = (
            indexed_df.withColumn("proto", proto_expr)
            .withColumn("service", service_expr)
            .withColumn("state", state_expr)
            .withColumn("attack_cat", attack_expr)
            .withColumn("dur", (F.rand(seed + 1) * 2.5))
            .withColumn("spkts", (F.rand(seed + 2) * 120).cast(LongType()) + 1)
            .withColumn("dpkts", (F.rand(seed + 3) * 150).cast(LongType()) + 1)
            .withColumn("sbytes", (F.col("spkts") * (F.rand(seed + 4) * 800 + 40)).cast(LongType()))
            .withColumn("dbytes", (F.col("dpkts") * (F.rand(seed + 5) * 1200 + 40)).cast(LongType()))
            .withColumn("rate", (F.col("sbytes") + F.col("dbytes")) / (F.col("dur") + 0.001))
            .withColumn("sttl", (F.rand(seed + 6) * 254 + 1).cast(IntegerType()))
            .withColumn("dttl", (F.rand(seed + 7) * 254 + 1).cast(IntegerType()))
            .withColumn("sload", F.col("rate") * 0.4)
            .withColumn("dload", F.col("rate") * 0.6)
            .withColumn("sloss", (F.col("spkts") * 0.05).cast(LongType()))
            .withColumn("dloss", (F.col("dpkts") * 0.04).cast(LongType()))
            .withColumn("sinpkt", F.rand(seed + 8) * 20.0)
            .withColumn("dinpkt", F.rand(seed + 9) * 20.0)
            .withColumn("sjit", F.rand(seed + 10) * 15.0)
            .withColumn("djit", F.rand(seed + 11) * 15.0)
            .withColumn("swin", (F.rand(seed + 12) * 255).cast(LongType()))
            .withColumn("stcpb", (F.rand(seed + 13) * 1000000).cast(LongType()))
            .withColumn("dtcpb", (F.rand(seed + 14) * 1000000).cast(LongType()))
            .withColumn("dwin", (F.rand(seed + 15) * 255).cast(LongType()))
            .withColumn("tcprtt", F.rand(seed + 16) * 0.05)
            .withColumn("synack", F.rand(seed + 17) * 0.02)
            .withColumn("ackdat", F.rand(seed + 18) * 0.03)
            .withColumn("smean", F.col("sbytes") / F.col("spkts"))
            .withColumn("dmean", F.col("dbytes") / F.col("dpkts"))
            .withColumn("trans_depth", (F.rand(seed + 19) * 2).cast(LongType()))
            .withColumn("response_body_len", (F.rand(seed + 20) * 2000).cast(LongType()))
            .withColumn("ct_srv_src", (F.rand(seed + 21) * 15 + 1).cast(IntegerType()))
            .withColumn("ct_state_ttl", (F.rand(seed + 22) * 6).cast(IntegerType()))
            .withColumn("ct_dst_ltm", (F.rand(seed + 23) * 15 + 1).cast(IntegerType()))
            .withColumn("ct_src_dport_ltm", (F.rand(seed + 24) * 10 + 1).cast(IntegerType()))
            .withColumn("ct_dst_sport_ltm", (F.rand(seed + 25) * 10 + 1).cast(IntegerType()))
            .withColumn("ct_dst_src_ltm", (F.rand(seed + 26) * 15 + 1).cast(IntegerType()))
            .withColumn("is_ftp_login", (F.rand(seed + 27) > 0.95).cast(IntegerType()))
            .withColumn("ct_ftp_cmd", (F.col("is_ftp_login") * (F.rand(seed + 28) * 2)).cast(IntegerType()))
            .withColumn("ct_flw_http_mthd", (F.rand(seed + 29) > 0.8).cast(IntegerType()))
            .withColumn("ct_src_ltm", (F.rand(seed + 30) * 20 + 1).cast(IntegerType()))
            .withColumn("ct_srv_dst", (F.rand(seed + 31) * 20 + 1).cast(IntegerType()))
            .withColumn("is_sm_ips_ports", (F.rand(seed + 32) > 0.98).cast(IntegerType()))
            .drop("proto_idx", "service_idx", "state_idx", "attack_idx")
        )

        metadata: Dict[str, Any] = {
            "is_synthetic": True,
            "benchmark_valid": False,
            "source": "SentinelSpark_Synthetic_Generator",
            "num_records": num_samples,
            "seed": seed,
            "schema_version": "UNSW-NB15-45col",
            "warning": (
                "DEMO MODE WARNING: Generated dataset is SYNTHETIC. "
                "Must NEVER be represented or published as official UNSW-NB15 benchmark results."
            ),
            "generated_at": datetime.now().isoformat(),
        }

        self.metadata = metadata
        normalized = self.normalize_schema(synthetic_df)
        return normalized, metadata

    def generate_synthetic_dataset(self, sample_size: int = 10000, seed: int = 42) -> DataFrame:
        """Backward-compatible helper returning single DataFrame for tests and legacy callers."""
        df, _ = self.generate_synthetic_data(num_samples=sample_size, seed=seed)
        return df
