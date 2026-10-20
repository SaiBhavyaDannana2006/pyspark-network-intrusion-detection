"""
Unit tests for SentinelSpark data ingestion and synthetic fallback.
"""

from pathlib import Path
import pytest
from src.data_loader import DataLoader, get_unsw_nb15_schema
from src.utils import load_config


def test_schema_definition():
    """Verify that the official UNSW-NB15 schema contains 45 columns with expected fields."""
    schema = get_unsw_nb15_schema()
    field_names = [f.name for f in schema.fields]
    assert "label" in field_names
    assert "attack_cat" in field_names
    assert "proto" in field_names
    assert "service" in field_names
    assert "state" in field_names
    assert "dur" in field_names
    assert "spkts" in field_names
    assert "dpkts" in field_names
    assert "sbytes" in field_names
    assert "dbytes" in field_names
    assert len(field_names) == 45


def test_synthetic_data_generation(spark):
    """Verify synthetic dataset generator conforms to UNSW-NB15 schema and distributions."""
    config = load_config()
    loader = DataLoader(spark, config)
    df = loader.generate_synthetic_dataset(sample_size=150)

    assert df.count() == 150
    assert "label" in df.columns
    assert "attack_cat" in df.columns
    assert "spkts" in df.columns

    # Verify both normal and attack instances exist
    labels = [r[0] for r in df.select("label").distinct().collect()]
    assert 0 in labels
    assert 1 in labels

    # Verify attack categories exist
    attack_cats = [r[0] for r in df.select("attack_cat").distinct().collect()]
    assert "Normal" in attack_cats
    assert len(attack_cats) > 1


def test_csv_loading_logic(spark, tmp_path):
    """Test CSV file ingestion, header canonicalization (trim and lowercase), and column parsing."""
    config = load_config()
    loader = DataLoader(spark, config)

    # Create a mock CSV with noisy headers (extra spaces, uppercase)
    csv_file = tmp_path / "test_unsw.csv"
    csv_content = (
        " ID , DUR , PROTO , SERVICE , STATE , SPKTS , DPKTS , SBYTES , DBYTES , LABEL , ATTACK_CAT \n"
        "1, 0.5, TCP , HTTP , FIN , 10, 8, 1200, 3400, 0, Normal\n"
        "2, 1.2, UDP , DNS , CON , 4, 4, 300, 500, 1, Exploits\n"
        "3, 0.05, TCP , - , INT , 100, 2, 45000, 80, 1, DoS\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    loaded_df = loader.load_csv(str(csv_file))
    assert loaded_df.count() == 3

    # Check canonicalized headers
    cols = loaded_df.columns
    assert "dur" in cols
    assert "proto" in cols
    assert "label" in cols
    assert "attack_cat" in cols
    assert "DUR" not in cols
    assert " PROTO " not in cols
