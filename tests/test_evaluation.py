"""
Unit tests for model evaluation, metrics computation, and confusion matrix validation.
"""

import pytest
from pyspark.ml.linalg import Vectors
from src.evaluation import ModelEvaluator
from src.utils import load_config


def test_binary_evaluation_metrics_and_confusion_matrix(spark):
    """Test binary precision, recall, accuracy, F1, FPR, FNR, and 2x2 confusion matrix."""
    config = load_config()
    evaluator = ModelEvaluator(config)

    # Mock predictions DataFrame:
    # 2 True Positives (1.0 -> 1.0)
    # 1 False Positive (0.0 -> 1.0)
    # 1 True Negative (0.0 -> 0.0)
    # 0 False Negatives (1.0 -> 0.0)
    data = [
        (1.0, 1.0, Vectors.dense([0.1, 0.9])),
        (0.0, 0.0, Vectors.dense([0.8, 0.2])),
        (1.0, 1.0, Vectors.dense([0.2, 0.8])),
        (0.0, 1.0, Vectors.dense([0.4, 0.6])),
    ]
    df = spark.createDataFrame(data, ["label", "prediction", "rawPrediction"])

    metrics = evaluator.evaluate_binary(df)

    # Core metrics
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f1_score" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "attack_precision" in metrics
    assert "attack_recall" in metrics
    assert "fpr" in metrics
    assert "fnr" in metrics

    # Mathematical assertions
    # Accuracy = (2 + 1) / 4 = 0.75
    assert abs(metrics["accuracy"] - 0.75) < 1e-4

    # Attack Precision = TP / (TP + FP) = 2 / (2 + 1) = 0.6667
    assert abs(metrics["attack_precision"] - (2.0 / 3.0)) < 1e-4

    # Attack Recall = TP / (TP + FN) = 2 / (2 + 0) = 1.0
    assert abs(metrics["attack_recall"] - 1.0) < 1e-4

    # False Positive Rate = FP / (FP + TN) = 1 / (1 + 1) = 0.5
    assert abs(metrics["fpr"] - 0.5) < 1e-4

    # False Negative Rate = FN / (FN + TP) = 0 / 2 = 0.0
    assert abs(metrics["fnr"] - 0.0) < 1e-4

    # Confusion matrix elements
    cm = metrics["confusion_matrix"]
    assert cm["tp"] == 2
    assert cm["fp"] == 1
    assert cm["tn"] == 1
    assert cm["fn"] == 0


def test_multiclass_evaluation_metrics_and_taxonomy(spark):
    """Test multiclass accuracy, macro-F1, weighted-F1, and per-class metrics."""
    config = load_config()
    evaluator = ModelEvaluator(config)

    # Mock multiclass predictions DataFrame
    data = [
        (0.0, 0.0, "Exploits", "Exploits"),
        (0.0, 0.0, "Exploits", "Exploits"),
        (1.0, 1.0, "DoS", "DoS"),
        (1.0, 2.0, "DoS", "Fuzzers"),
        (2.0, 2.0, "Fuzzers", "Fuzzers"),
    ]
    df = spark.createDataFrame(
        data,
        ["indexed_attack_cat", "stage2_prediction", "attack_cat", "predicted_attack_cat"]
    )

    metrics = evaluator.evaluate_multiclass(
        df,
        label_col="indexed_attack_cat",
        pred_col="stage2_prediction",
        target_name_col="attack_cat",
        pred_name_col="predicted_attack_cat",
    )

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_class_metrics" in metrics
    assert "confusion_matrix" in metrics

    # Exploits has 2/2 correct (Recall=1.0, Precision=1.0)
    exploits_m = metrics["per_class_metrics"]["Exploits"]
    assert exploits_m["recall"] == 1.0
    assert exploits_m["precision"] == 1.0

    # Total accuracy = 4 / 5 = 0.8
    assert abs(metrics["accuracy"] - 0.8) < 1e-4
