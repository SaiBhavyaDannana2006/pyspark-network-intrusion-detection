"""
End-to-end pipeline smoke test on a tiny synthetic dataset fixture (200 records).
"""

import pytest
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.feature_pipeline import FeaturePipelineBuilder
from src.binary_detection import BinaryDetectionFactory
from src.evaluation import ModelEvaluator
from src.utils import load_config


def test_pipeline_smoke(spark):
    """Execute complete end-to-end SentinelSpark pipeline workflow on a 200-row fixture."""
    config = load_config()

    # 1. Ingestion: Generate 200-row synthetic fixture
    loader = DataLoader(spark, config)
    raw_df = loader.generate_synthetic_dataset(sample_size=200)
    assert raw_df.count() == 200
    assert "label" in raw_df.columns
    assert "attack_cat" in raw_df.columns

    # 2. Preprocessing: Clean, deduplicate, and split 70/15/15
    preprocessor = Preprocessor(config)
    cleaned_df = preprocessor.clean(raw_df)
    train_df, val_df, test_df = preprocessor.split(cleaned_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert train_df.count() > 0
    assert test_df.count() > 0

    train_proc = preprocessor.fit_transform_train(train_df)
    test_proc = preprocessor.transform_eval(test_df)

    # 3. Feature Engineering: Derive behavioral network ratios
    engineer = FeatureEngineer(config)
    train_feat = engineer.engineer_features(train_proc)
    test_feat = engineer.engineer_features(test_proc)

    assert "packet_rate" in train_feat.columns
    assert "byte_rate" in train_feat.columns
    assert "avg_bytes_per_packet" in train_feat.columns

    # 4. Feature Pipeline: Build and fit Spark MLlib feature pipeline
    builder = FeaturePipelineBuilder(config)
    pipeline, feature_cols = builder.build_pipeline(target_mode="binary")
    pipeline_model = pipeline.fit(train_feat)

    train_vec = pipeline_model.transform(train_feat)
    test_vec = pipeline_model.transform(test_feat)

    assert "features" in train_vec.columns
    assert "features" in test_vec.columns

    # 5. Model Training: Train fast Stage 1 Logistic Regression classifier
    factory = BinaryDetectionFactory(config)
    lr = factory.get_logistic_regression()
    lr.setMaxIter(10)
    lr_model = lr.fit(train_vec)
    assert lr_model is not None

    # 6. Prediction & Evaluation: Predict on test split and compute metrics
    test_preds = lr_model.transform(test_vec)
    assert "prediction" in test_preds.columns
    assert "probability" in test_preds.columns

    evaluator = ModelEvaluator(config)
    metrics = evaluator.evaluate_binary(test_preds)

    assert "accuracy" in metrics
    assert "attack_recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
