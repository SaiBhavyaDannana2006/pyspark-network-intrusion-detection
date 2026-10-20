"""
Unit tests for data cleaning, missing value imputation, and zero-leakage outlier capping.
"""

from pathlib import Path
import pytest
from pyspark.sql import functions as F
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.utils import load_config


def test_preprocessing_clean(spark):
    """Verify preprocessor cleaning eliminates nulls across continuous features."""
    config = load_config()
    loader = DataLoader(spark, config)
    raw_df = loader.generate_synthetic_dataset(sample_size=50)

    preprocessor = Preprocessor(config)
    cleaned_df = preprocessor.clean(raw_df)

    assert cleaned_df.count() == 50
    # Ensure no nulls in numeric continuous features
    null_count = cleaned_df.filter(cleaned_df.dur.isNull()).count()
    assert null_count == 0


def test_duplicate_removal_with_tracking(spark):
    """Verify exact duplicate rows are removed and tracked accurately."""
    config = load_config()
    loader = DataLoader(spark, config)
    raw_df = loader.generate_synthetic_dataset(sample_size=40)

    # Artificially duplicate 10 rows
    dupes = raw_df.limit(10)
    with_dupes = raw_df.unionByName(dupes)
    assert with_dupes.count() == 50

    preprocessor = Preprocessor(config)
    deduped = preprocessor.remove_duplicates(with_dupes)

    assert deduped.count() == 40
    assert preprocessor.duplicate_count == 10


def test_categorical_null_and_unknown_handling(spark):
    """Verify categorical nulls, empty strings, and variant categories are sanitized."""
    config = load_config()
    loader = DataLoader(spark, config)
    raw_df = loader.generate_synthetic_dataset(sample_size=20)

    # Inject nulls and empty strings into categoricals
    dirty_df = (
        raw_df
        .withColumn("proto", F.when(F.col("id") == 0, F.lit(None)).otherwise(F.col("proto")))
        .withColumn("service", F.when(F.col("id") == 1, F.lit("")).otherwise(F.col("service")))
        .withColumn("state", F.when(F.col("id") == 2, F.lit("null")).otherwise(F.col("state")))
        .withColumn("attack_cat", F.when(F.col("id") == 3, F.lit("Backdoors")).otherwise(F.col("attack_cat")))
    )

    preprocessor = Preprocessor(config)
    cleaned = preprocessor.handle_categoricals(dirty_df)

    # Proto should be 'unknown' for id 0
    proto_val = cleaned.filter(F.col("id") == 0).select("proto").first()[0]
    assert proto_val == "unknown"

    # Service should be 'unknown' for id 1
    service_val = cleaned.filter(F.col("id") == 1).select("service").first()[0]
    assert service_val == "unknown"

    # State should be 'unknown' for id 2
    state_val = cleaned.filter(F.col("id") == 2).select("state").first()[0]
    assert state_val == "unknown"

    # Attack cat 'Backdoors' should be standardized to 'Backdoor'
    cat_val = cleaned.filter(F.col("id") == 3).select("attack_cat").first()[0]
    assert cat_val == "Backdoor"


def test_missing_value_imputation(spark):
    """Verify numeric missing values are imputed with training-fitted medians."""
    config = load_config()
    loader = DataLoader(spark, config)
    raw_df = loader.generate_synthetic_dataset(sample_size=60)

    # Inject nulls into numeric column 'dur'
    with_nulls = raw_df.withColumn(
        "dur", F.when(F.col("id") < 5, F.lit(None)).otherwise(F.col("dur"))
    )
    assert with_nulls.filter(F.col("dur").isNull()).count() == 5

    preprocessor = Preprocessor(config)
    train_proc = preprocessor.fit_transform_train(with_nulls)

    # Imputed result should have 0 nulls in 'dur'
    assert train_proc.filter(F.col("dur").isNull()).count() == 0
    assert "dur" in preprocessor.fitted_medians


def test_zero_leakage_imputation_and_outlier_capping(spark, tmp_path):
    """Verify training split parameters are saved and applied to validation/test without leakage."""
    config = load_config()
    loader = DataLoader(spark, config)
    raw_df = loader.generate_synthetic_dataset(sample_size=100)

    preprocessor = Preprocessor(config)
    train_df, val_df, test_df = preprocessor.split(raw_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # Fit exclusively on train
    train_proc = preprocessor.fit_transform_train(train_df)
    assert train_proc.count() == train_df.count()
    assert len(preprocessor.fitted_medians) > 0
    assert len(preprocessor.outlier_bounds) > 0

    # Transform val and test using fitted parameters
    val_proc = preprocessor.transform_eval(val_df)
    test_proc = preprocessor.transform_eval(test_df)

    assert val_proc.count() == val_df.count()
    assert test_proc.count() == test_df.count()

    # Verify no records dropped (especially attack instances preserved)
    orig_attacks = val_df.filter(F.col("label") == 1).count()
    proc_attacks = val_proc.filter(F.col("label") == 1).count()
    assert orig_attacks == proc_attacks

    # Verify parameters can be serialized for audit trail
    param_file = str(tmp_path / "fitted_prep_params.json")
    preprocessor.save_fitted_parameters(param_file)
    params = preprocessor.get_fitted_parameters()
    assert "fitted_medians" in params
    assert "outlier_bounds" in params
