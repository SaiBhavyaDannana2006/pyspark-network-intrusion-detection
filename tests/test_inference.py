"""
Unit tests for SentinelSpark Persistence & Inference Engine.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.inference import InferenceEngine, save_pipeline_model, save_run_artifacts
from src.utils import load_config


def test_save_run_artifacts(tmp_path):
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"

    metadata = {"runtime_seconds": 12.3, "stage1_best_model": "RandomForest"}
    config_snapshot = {"app_name": "Test"}
    metrics = {"binary_test": {"accuracy": 0.95, "f1_score": 0.94}}
    features = ["dur", "spkts", "dpkts"]

    save_run_artifacts(
        models_dir=models_dir,
        results_dir=results_dir,
        metadata=metadata,
        config_snapshot=config_snapshot,
        metrics=metrics,
        feature_mapping=features,
    )

    assert (models_dir / "training_run_metadata.json").exists()
    assert (models_dir / "config_snapshot.json").exists()
    assert (models_dir / "feature_mapping.json").exists()
    assert (results_dir / "metrics.json").exists()


def test_inference_engine_batch_and_single_flow(spark):
    config = load_config()
    engine = InferenceEngine(config, spark=spark)

    # 1. Batch prediction on pandas DataFrame
    pdf = pd.DataFrame([
        {
            "id": 1,
            "dur": 0.5,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "spkts": 10,
            "dpkts": 8,
            "sbytes": 1000,
            "dbytes": 800,
            "rate": 36.0,
            "sttl": 64,
            "dttl": 64,
            "sload": 16000.0,
            "dload": 12800.0,
            "sloss": 0,
            "dloss": 0,
            "sinpkt": 50.0,
            "dinpkt": 62.5,
            "sjit": 0.0,
            "djit": 0.0,
            "swin": 255,
            "stcpb": 1000,
            "dtcpb": 2000,
            "dwin": 255,
            "tcprtt": 0.05,
            "synack": 0.02,
            "ackdat": 0.03,
            "smean": 100.0,
            "dmean": 100.0,
            "trans_depth": 1,
            "response_body_len": 0,
            "ct_srv_src": 1,
            "ct_state_ttl": 0,
            "ct_dst_ltm": 1,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "ct_dst_src_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 1,
            "ct_src_ltm": 1,
            "ct_srv_dst": 1,
            "is_sm_ips_ports": 0,
        }
    ])

    batch_preds = engine.predict(pdf)
    assert not batch_preds.empty
    assert "attack_status" in batch_preds.columns
    assert "attack_probability" in batch_preds.columns
    assert "predicted_attack_cat" in batch_preds.columns
    assert batch_preds.iloc[0]["attack_status"] in ("ATTACK", "NORMAL")

    # 2. Single flow prediction
    single_res = engine.predict_single_flow(pdf.iloc[0].to_dict())
    assert isinstance(single_res, dict)
    assert "is_attack" in single_res
    assert "attack_status" in single_res
    assert "attack_probability" in single_res
    assert "predicted_attack_cat" in single_res
    assert single_res["attack_status"] in ("ATTACK", "NORMAL")
