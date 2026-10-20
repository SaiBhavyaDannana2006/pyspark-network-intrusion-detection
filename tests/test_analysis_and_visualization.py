"""
Tests for Feature Importance, Attack Analysis, and Visualizer 8-Chart Generation.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.attack_analysis import AttackAnalyzer
from src.data_loader import DataLoader
from src.feature_importance import FeatureImportanceExtractor
from src.spark_session import get_spark_session, stop_spark_session
from src.utils import load_config
from src.visualization import Visualizer





@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def sample_df(spark, config):
    loader = DataLoader(spark, config)
    df, _ = loader.generate_synthetic_data(num_samples=200)
    return df


def test_attack_analyzer_methods(sample_df, config):
    analyzer = AttackAnalyzer(config)

    # 1. Protocol attack rates
    proto_df = analyzer.analyze_protocol_attack_rates(sample_df, top_n=5)
    assert not proto_df.empty
    assert "proto" in proto_df.columns
    assert "attack_rate" in proto_df.columns

    # 2. Service attack rates
    service_df = analyzer.analyze_service_attack_rates(sample_df, top_n=5)
    assert not service_df.empty
    assert "service" in service_df.columns
    assert "attack_rate" in service_df.columns

    # 3. Traffic statistics per attack class
    traffic_stats = analyzer.compute_traffic_stats_per_class(sample_df)
    assert not traffic_stats.empty
    assert "attack_cat" in traffic_stats.columns
    assert "flow_count" in traffic_stats.columns

    # 4. Confused categories
    mock_preds = sample_df.withColumn("predicted_attack_cat", sample_df.attack_cat)
    confused_df = analyzer.find_top_confused_categories(mock_preds, top_n=5)
    assert isinstance(confused_df, pd.DataFrame)


def test_visualizer_eight_charts_generation(config, tmp_path):
    config["models"]["results_dir"] = str(tmp_path / "figures")
    visualizer = Visualizer(config)

    # 1. normal_vs_attack_dist.png
    target_df = pd.DataFrame({
        "class_name": ["Normal (Benign)", "Intrusion (Malicious)"],
        "count": [600, 400]
    })
    p1 = visualizer.plot_normal_vs_attack_distribution(target_df)
    assert Path(p1).exists()
    assert Path(p1).name == "normal_vs_attack_dist.png"

    # 2. attack_category_dist.png
    cat_df = pd.DataFrame({
        "attack_cat": ["Normal", "Generic", "Exploits", "Fuzzers", "DoS"],
        "count": [600, 150, 120, 80, 50]
    })
    p2 = visualizer.plot_attack_category_distribution(cat_df)
    assert Path(p2).exists()
    assert Path(p2).name == "attack_category_dist.png"

    # 3. model_comparison_metrics.png
    metrics_map = {
        "LogisticRegression": {"attack_recall": 0.88, "f1_score": 0.89, "attack_precision": 0.90, "pr_auc": 0.92, "roc_auc": 0.94},
        "RandomForest": {"attack_recall": 0.94, "f1_score": 0.95, "attack_precision": 0.96, "pr_auc": 0.97, "roc_auc": 0.98},
        "GBT": {"attack_recall": 0.93, "f1_score": 0.94, "attack_precision": 0.95, "pr_auc": 0.96, "roc_auc": 0.97},
    }
    p3 = visualizer.plot_model_comparison_metrics(metrics_map)
    assert Path(p3).exists()
    assert Path(p3).name == "model_comparison_metrics.png"

    # 4. binary_confusion_matrix.png
    cm_dict = {"tp": 380, "fp": 20, "tn": 570, "fn": 30}
    p4 = visualizer.plot_binary_confusion_matrix(cm_dict)
    assert Path(p4).exists()
    assert Path(p4).name == "binary_confusion_matrix.png"

    # 5. multiclass_confusion_matrix.png
    cm_data = pd.DataFrame(
        [[45, 2, 3], [1, 38, 1], [4, 0, 36]],
        index=["Exploits", "DoS", "Fuzzers"],
        columns=["Exploits", "DoS", "Fuzzers"]
    )
    p5 = visualizer.plot_multiclass_confusion_matrix(cm_data)
    assert Path(p5).exists()
    assert Path(p5).name == "multiclass_confusion_matrix.png"

    # 6. feature_importance_top20.png
    imp_df = pd.DataFrame({
        "feature": [f"feature_{i}" for i in range(25)],
        "importance": [round(0.25 / (i + 1), 4) for i in range(25)]
    })
    p6 = visualizer.plot_feature_importance_top20(imp_df)
    assert Path(p6).exists()
    assert Path(p6).name == "feature_importance_top20.png"

    # 7. per_class_f1_scores.png
    f1_map = {
        "Exploits": {"f1": 0.91},
        "DoS": {"f1": 0.88},
        "Fuzzers": {"f1": 0.82},
        "Worms": {"f1": 0.45},
    }
    p7 = visualizer.plot_per_class_f1_scores(f1_map)
    assert Path(p7).exists()
    assert Path(p7).name == "per_class_f1_scores.png"

    # 8. protocol_attack_rate.png
    proto_df = pd.DataFrame({
        "proto": ["tcp", "udp", "unas", "ospf"],
        "attack_rate": [72.5, 41.2, 85.0, 10.0]
    })
    p8 = visualizer.plot_protocol_attack_rate(proto_df)
    assert Path(p8).exists()
    assert Path(p8).name == "protocol_attack_rate.png"
