"""
Unit tests for network behavioral feature engineering.
"""

import pytest
from pyspark.sql import functions as F
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.spark_session import SparkSessionManager
from src.utils import load_config





def test_feature_engineering_columns(spark):
    config = load_config()
    loader = DataLoader(spark, config)
    raw_df = loader.generate_synthetic_dataset(sample_size=30)
    cleaned_df = Preprocessor(config).clean(raw_df)

    engineer = FeatureEngineer(config)
    feat_df = engineer.engineer_features(cleaned_df)

    expected_cols = [
        "bytes_per_packet_src",
        "bytes_per_packet_dst",
        "packet_ratio",
        "byte_ratio",
        "tcp_handshake_ratio",
        "total_packets",
        "total_bytes",
        "packet_rate",
        "byte_rate",
        "avg_bytes_per_packet",
        "sbyte_ratio",
        "packet_size_ratio",
        "load_ratio",
        "ttl_ratio",
        "tcprtt_ratio",
    ]
    for col in expected_cols:
        assert col in feat_df.columns


def test_safe_division_with_zero_duration(spark):
    config = load_config()
    loader = DataLoader(spark, config)
    raw_df = loader.generate_synthetic_dataset(sample_size=20)

    # Force duration and packets to 0 to test safe division handling
    zero_df = raw_df.withColumn("dur", F.lit(0.0)).withColumn("spkts", F.lit(0)).withColumn("dpkts", F.lit(0))

    engineer = FeatureEngineer(config)
    feat_df = engineer.engineer_features(zero_df)

    # Verify no infinite or NaN values in rate features
    nan_rates = feat_df.filter(F.isnan("packet_rate") | F.col("packet_rate").isNull()).count()
    assert nan_rates == 0

    nan_byte_rate = feat_df.filter(F.isnan("byte_rate") | F.col("byte_rate").isNull()).count()
    assert nan_byte_rate == 0


def test_target_leakage_protection_and_stage2_isolation(spark):
    config = load_config()
    loader = DataLoader(spark, config)
    raw_df = loader.generate_synthetic_dataset(sample_size=50)

    engineer = FeatureEngineer(config)

    # 1. Non-predictive column dropping
    with_index = raw_df.withColumn("index", F.lit(123)).withColumn("_c0", F.lit(456))
    sanitized = engineer.drop_non_predictive_columns(with_index)
    assert "id" not in sanitized.columns
    assert "index" not in sanitized.columns
    assert "_c0" not in sanitized.columns

    # 2. Stage 2 target isolation (removes Normal records)
    stage2_data = engineer.filter_stage2_threats(raw_df)
    labels = [r[0] for r in stage2_data.select("label").distinct().collect()]
    attack_cats = [r[0] for r in stage2_data.select("attack_cat").distinct().collect()]

    assert all(l == 1 for l in labels)
    assert "Normal" not in attack_cats
