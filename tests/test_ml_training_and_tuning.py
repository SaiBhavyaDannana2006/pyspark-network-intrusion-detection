"""
Tests for Stage 1 Binary Detection, Stage 2 Multiclass Profiling,
Hyperparameter Tuning, and Training Orchestrator.
"""

from pathlib import Path
import pytest
from src.attack_profiling import AttackProfilingEngine
from src.binary_detection import BinaryDetectionEngine, BinaryDetectionFactory
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.feature_pipeline import FeaturePipelineBuilder
from src.hyperparameter_tuning import HyperparameterTuner
from src.model_training import ModelTrainer
from src.preprocessing import Preprocessor
from src.spark_session import get_spark_session, stop_spark_session
from src.utils import load_config





@pytest.fixture(scope="session")
def config():
    cfg = load_config()
    # Micro size for fast testing
    cfg["data"]["synthetic_fallback"]["num_samples"] = 250
    # Micro hyperparameters for fast test runs
    cfg["models"]["stage1_binary"]["random_forest"]["num_trees"] = 5
    cfg["models"]["stage1_binary"]["random_forest"]["max_depth"] = 3
    cfg["models"]["stage1_binary"]["gbt"]["max_iter"] = 5
    cfg["models"]["stage1_binary"]["gbt"]["max_depth"] = 3
    cfg["models"]["stage1_binary"]["logistic_regression"]["max_iter"] = 10
    cfg["models"]["stage2_multiclass"]["random_forest"]["num_trees"] = 5
    cfg["models"]["stage2_multiclass"]["random_forest"]["max_depth"] = 3
    return cfg


@pytest.fixture(scope="session")
def preprocessed_splits(spark, config):
    loader = DataLoader(spark, config)
    raw_df, _ = loader.generate_synthetic_data(num_samples=250)

    preprocessor = Preprocessor(config)
    cleaned_df = preprocessor.clean(raw_df)
    train_raw, val_raw, test_raw = preprocessor.split(cleaned_df)

    train_df = preprocessor.fit_transform_train(train_raw)
    val_df = preprocessor.transform_eval(val_raw)
    test_df = preprocessor.transform_eval(test_raw)

    engineer = FeatureEngineer(config)
    train_eng = engineer.transform(train_df)
    val_eng = engineer.transform(val_df)
    test_eng = engineer.transform(test_df)
    engineered_cols = engineer.get_engineered_feature_names()

    pipe_builder = FeaturePipelineBuilder(config)
    # Include multiclass target indexer so both stages have their features ready
    pipe_model = pipe_builder.fit(
        train_eng,
        engineered_cols=engineered_cols,
        include_multiclass_target_indexer=False
    )
    train_vec = pipe_model.transform(train_eng).cache()
    val_vec = pipe_model.transform(val_eng).cache()
    test_vec = pipe_model.transform(test_eng).cache()

    return train_vec, val_vec, test_vec


def test_stage1_binary_models_comparison_and_selection(preprocessed_splits, config):
    train_vec, val_vec, _ = preprocessed_splits
    engine = BinaryDetectionEngine(config)

    result = engine.train_and_evaluate_all(train_vec, val_vec)

    assert result.best_model is not None
    assert result.best_model_name in ("LogisticRegression", "RandomForestClassifier", "GBTClassifier")
    assert "LogisticRegression" in result.all_models
    assert "RandomForestClassifier" in result.all_models
    assert "GBTClassifier" in result.all_models

    # Verify metrics structure
    for name, metrics in result.all_metrics.items():
        assert "attack_recall" in metrics
        assert "f1_score" in metrics
        assert "selection_score" in metrics
        assert 0.0 <= metrics["attack_recall"] <= 1.0


def test_stage2_multiclass_attack_profiling_and_hierarchical_routing(preprocessed_splits, config):
    train_vec, val_vec, _ = preprocessed_splits
    engine = AttackProfilingEngine(config)

    # 1. Train Stage 2 Profiler strictly on attacks
    stage2_result = engine.train(train_vec)
    assert stage2_result.pipeline_model is not None
    assert len(stage2_result.labels) > 0
    assert "Normal" not in stage2_result.labels, "Stage 2 classifier must not contain Normal category!"

    # 2. Test hierarchical routing: evaluate Stage 2 only on Stage 1 flagged threats
    binary_engine = BinaryDetectionEngine(config)
    binary_model = binary_engine.factory.get_logistic_regression(use_weights=False).fit(train_vec)
    stage1_preds = binary_model.transform(val_vec)

    hierarchical_preds, stage2_metrics = engine.evaluate_hierarchical(stage1_preds, stage2_result)
    assert "predicted_attack_cat" in hierarchical_preds.columns

    # Verify that records cleared as Normal by Stage 1 have predicted_attack_cat == 'Normal'
    normal_preds = hierarchical_preds.filter(hierarchical_preds.prediction == 0.0)
    normal_distinct_cats = [r[0] for r in normal_preds.select("predicted_attack_cat").distinct().collect()]
    assert normal_distinct_cats == ["Normal"], "Stage 1 cleared flows must be labeled 'Normal'!"


def test_hyperparameter_tuning_toggleable(preprocessed_splits, config):
    train_vec, _, _ = preprocessed_splits
    tuner = HyperparameterTuner(config)
    factory = BinaryDetectionFactory(config)
    base_rf = factory.get_random_forest()

    # Test fast-path when disabled
    tuner.is_enabled = False
    result_disabled = tuner.tune_random_forest(base_rf, train_vec)
    assert result_disabled.is_tuned is False
    assert result_disabled.best_model is not None

    # Test tuning execution when enabled with micro-grid
    tuner.is_enabled = True
    tuner.tuning_cfg["param_grids"] = {
        "rf_num_trees": [3, 5],
        "rf_max_depth": [3],
        "rf_max_bins": [128],
    }
    result_enabled = tuner.tune_random_forest(base_rf, train_vec)
    assert result_enabled.is_tuned is True
    assert result_enabled.best_model is not None
    assert "numTrees" in result_enabled.best_params


def test_model_training_orchestrator(preprocessed_splits, config, tmp_path):
    train_vec, val_vec, _ = preprocessed_splits
    # Override artifacts dir to temp path
    config["models"]["artifacts_dir"] = str(tmp_path / "models")
    trainer = ModelTrainer(config)

    artifacts = trainer.train_full_pipeline(train_vec, val_vec)

    assert artifacts.stage1_best_model is not None
    assert artifacts.stage2_profiling.pipeline_model is not None
    assert artifacts.hierarchical_val_predictions is not None
    assert "stage1" in artifacts.metadata
    assert "stage2" in artifacts.metadata

    # Verify persisted metadata and models
    meta_file = Path(config["models"]["artifacts_dir"]) / "training_run_metadata.json"
    assert meta_file.exists()
