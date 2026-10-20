"""
==============================================================================
SentinelSpark — Core Utilities & Configuration Loader
==============================================================================
Provides resilient configuration management, logging setup, path resolution,
and execution timing decorators.
==============================================================================
"""

import functools
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import yaml

# Base directory for relative resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logger(
    name: str = "SentinelSpark",
    level: int = logging.INFO,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a structured logger."""
    if log_format is None:
        log_format = (
            "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] - %(message)s"
        )
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration YAML file with fallback to default project config.

    Args:
        config_path: Path to YAML config file. If None, uses config/config.yaml

    Returns:
        Dict[str, Any]: Parsed configuration mapping.
    """
    if config_path is None:
        resolved_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        resolved_path = Path(config_path)

    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    if not resolved_path.exists():
        logger.warning(
            f"Config file not found at {resolved_path}. Initializing default fallback config."
        )
        return _get_default_config()

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            logger.debug(f"Loaded configuration from {resolved_path}")
            return cfg
    except Exception as e:
        logger.error(f"Error reading YAML from {resolved_path}: {e}")
        return _get_default_config()


def _get_default_config() -> Dict[str, Any]:
    """Provide hardcoded default configuration in case YAML cannot be read."""
    return {
        "spark": {
            "app_name": "SentinelSpark-Network-Intrusion-Intelligence",
            "master": "local[*]",
            "driver_memory": "4g",
            "executor_memory": "4g",
            "sql_shuffle_partitions": 8,
            "serializer": "org.apache.spark.serializer.KryoSerializer",
            "log_level": "WARN",
        },
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "models_dir": "models",
            "results_dir": "results",
            "figures_dir": "results/figures",
            "allow_synthetic_fallback": True,
            "synthetic_sample_size": 10000,
        },
        "splits": {
            "train_ratio": 0.70,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "seed": 42,
        },
        "features": {
            "id_column": "id",
            "target_binary": "label",
            "target_multiclass": "attack_cat",
        },
    }


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and log execution duration of pipeline components."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        logger.info(f"==> Starting execution: [{func.__name__}]")
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            logger.info(
                f"<== Completed: [{func.__name__}] in {duration:.2f} seconds ({duration/60:.2f} min)"
            )
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(
                f"[FAILED] [{func.__name__}] failed after {duration:.2f}s with error: {e}"
            )
            raise

    return wrapper


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists on the filesystem and return resolved Path object."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json_metrics(metrics: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Write metrics dictionary to JSON file with indentation."""
    target_path = Path(output_path)
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path
    ensure_directory(target_path.parent)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics output to {target_path}")
