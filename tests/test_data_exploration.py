"""
Tests for DataExplorer distributed profiling using native PySpark operations.
"""

from pathlib import Path
import pytest
from src.data_exploration import DataExplorer
from src.data_loader import DataLoader
from src.spark_session import get_spark_session, stop_spark_session
from src.utils import load_config





@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def sample_df(spark, config):
    loader = DataLoader(spark, config)
    df, _ = loader.generate_synthetic_data(num_samples=200)
    return df


def test_schema_overview(sample_df, config):
    explorer = DataExplorer(sample_df, config)
    overview, schema_df = explorer.get_schema_overview()

    assert overview["total_records"] == 200
    assert overview["total_columns"] >= 45
    assert not schema_df.empty
    assert "column_name" in schema_df.columns
    assert "data_type" in schema_df.columns


def test_audit_missing_and_empty(sample_df, config):
    explorer = DataExplorer(sample_df, config)
    audit_dict, audit_df = explorer.audit_missing_and_empty()

    assert isinstance(audit_dict, dict)
    assert not audit_df.empty
    assert "null_count" in audit_df.columns
    assert "empty_string_count" in audit_df.columns
    assert "dash_placeholder_count" in audit_df.columns


def test_target_distribution(sample_df, config):
    explorer = DataExplorer(sample_df, config)
    target_dict, target_df = explorer.analyze_target_distribution()

    assert "total_records" in target_dict
    assert not target_df.empty
    assert "label" in target_df.columns
    assert "ratio" in target_df.columns
    # Check that ratio sums approximately to 1.0
    total_ratio = target_df["ratio"].sum()
    assert abs(total_ratio - 1.0) < 0.05


def test_multiclass_distribution(sample_df, config):
    explorer = DataExplorer(sample_df, config)
    multi_dict, multi_df = explorer.analyze_multiclass_distribution()

    assert "attack_categories" in multi_dict
    assert not multi_df.empty
    assert "attack_cat" in multi_df.columns
    assert "percentage" in multi_df.columns


def test_compute_numeric_summary(sample_df, config):
    explorer = DataExplorer(sample_df, config)
    summary_dict, summary_df = explorer.compute_numeric_summary(columns=["dur", "sbytes", "dbytes", "rate"])

    assert not summary_df.empty
    assert "summary" in summary_df.columns
    summaries = summary_df["summary"].tolist()
    assert "mean" in summaries
    assert "stddev" in summaries
    assert "min" in summaries
    assert "max" in summaries


def test_categorical_distributions(sample_df, config):
    explorer = DataExplorer(sample_df, config)
    cat_dfs = explorer.analyze_categorical_distributions(top_n=5)

    assert "proto" in cat_dfs
    assert "service" in cat_dfs
    assert "state" in cat_dfs
    for name, c_df in cat_dfs.items():
        assert not c_df.empty
        assert "count" in c_df.columns
        assert "percentage" in c_df.columns


def test_run_full_exploration_with_export(sample_df, config, tmp_path):
    explorer = DataExplorer(sample_df, config)
    export_dir = str(tmp_path / "exploration_output")
    results = explorer.run_full_exploration(export_dir=export_dir)

    assert "overview" in results
    assert "missing_audit" in results
    assert "binary_target" in results
    assert "multiclass_target" in results
    assert "numeric_summary" in results

    # Verify CSV files created
    p = Path(export_dir)
    assert (p / "schema_overview.csv").exists()
    assert (p / "missing_values_audit.csv").exists()
    assert (p / "binary_target_distribution.csv").exists()
    assert (p / "multiclass_category_distribution.csv").exists()
    assert (p / "numeric_summary_statistics.csv").exists()
