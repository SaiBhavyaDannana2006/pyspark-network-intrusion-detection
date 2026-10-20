"""
==============================================================================
SentinelSpark — Spark Session Manager
==============================================================================
Thread-safe Singleton SparkSession factory for high-throughput distributed
ML on UNSW-NB15 network intrusion traffic.

Features:
- Thread-safe Singleton instance tracking
- Dynamic configuration via YAML dictionary or fallback defaults
- Local master fallback and memory ceiling enforcement
- JVM Log4j / SparkContext log suppression (WARN / ERROR)
- Robust lifecycle management, context manager support, and safe termination
==============================================================================
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Dict, Optional
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

# Guarantee PySpark worker resolves active Python interpreter on Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

logger = logging.getLogger("SentinelSpark.Session")


class SparkSessionManager:
    """Thread-safe Singleton Manager for Apache Spark Session.

    Provides controlled initialization, unified configuration propagation,
    JVM log suppression, and resource teardown for the SentinelSpark pipeline.
    """

    _instance: Optional["SparkSessionManager"] = None
    _lock: threading.Lock = threading.Lock()
    _spark: Optional[SparkSession] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "SparkSessionManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SparkSessionManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_active_session(cls) -> Optional[SparkSession]:
        """Return currently active SparkSession if available."""
        with cls._lock:
            if cls._spark is not None:
                return cls._spark
            return SparkSession.getActiveSession()

    @classmethod
    def get_or_create(
        cls,
        config: Optional[Dict[str, Any]] = None,
        app_name: Optional[str] = None,
        master: Optional[str] = None,
        driver_memory: Optional[str] = None,
        executor_memory: Optional[str] = None,
        log_level: Optional[str] = None,
    ) -> SparkSession:
        """Get an existing SparkSession or initialize a new one with the given configuration.

        Args:
            config: Optional dictionary containing nested spark configuration
            app_name: Override application name
            master: Override master URL (defaults to local[*] with fallback)
            driver_memory: Memory allocated to driver process (e.g., '4g')
            executor_memory: Memory allocated to executors (e.g., '4g')
            log_level: JVM logging level ('ALL', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'OFF')

        Returns:
            SparkSession: Configured, active PySpark session.
        """
        with cls._lock:
            if cls._spark is not None:
                try:
                    # Validate if existing session context is alive
                    if not cls._spark.sparkContext._jsc.sc().isStopped():
                        return cls._spark
                except Exception:
                    cls._spark = None

            spark_cfg = config.get("spark", {}) if config else {}

            resolved_app_name = (
                app_name
                or os.environ.get("SPARK_APP_NAME")
                or spark_cfg.get("app_name", "SentinelSpark-Network-Intrusion-Intelligence")
            )
            resolved_master = (
                master
                or os.environ.get("SPARK_MASTER")
                or spark_cfg.get("master", "local[*]")
            )
            resolved_driver_mem = (
                driver_memory
                or os.environ.get("SPARK_DRIVER_MEMORY")
                or spark_cfg.get("driver_memory", "4g")
            )
            resolved_exec_mem = (
                executor_memory
                or os.environ.get("SPARK_EXECUTOR_MEMORY")
                or spark_cfg.get("executor_memory", "4g")
            )
            resolved_log_level = (
                log_level
                or os.environ.get("SPARK_LOG_LEVEL")
                or spark_cfg.get("log_level", "WARN")
            )

            shuffle_partitions = spark_cfg.get("sql_shuffle_partitions", 8)
            serializer = spark_cfg.get(
                "serializer", "org.apache.spark.serializer.KryoSerializer"
            )
            driver_max_result = spark_cfg.get("driver_max_result_size", "2g")
            buffer_max = spark_cfg.get("kryoserializer_buffer_max", "512m")
            ui_enabled = spark_cfg.get("ui_enabled", False)

            # Build SparkConf
            conf = (
                SparkConf()
                .setAppName(resolved_app_name)
                .setMaster(resolved_master)
                .set("spark.pyspark.python", sys.executable)
                .set("spark.pyspark.driver.python", sys.executable)
                .set("spark.driver.memory", str(resolved_driver_mem))
                .set("spark.executor.memory", str(resolved_exec_mem))
                .set("spark.driver.maxResultSize", str(driver_max_result))
                .set("spark.sql.shuffle.partitions", str(shuffle_partitions))
                .set("spark.serializer", str(serializer))
                .set("spark.kryoserializer.buffer.max", str(buffer_max))
                .set("spark.sql.execution.arrow.pyspark.enabled", "true")
                .set("spark.ui.enabled", "true" if ui_enabled else "false")
                .set("spark.ui.showConsoleProgress", "false")
            )

            # Attempt building SparkSession with master fallback
            try:
                builder = SparkSession.builder.config(conf=conf)
                session = builder.getOrCreate()
            except Exception as primary_exc:
                logger.warning(
                    f"Failed to initialize Spark with master '{resolved_master}'. "
                    f"Falling back to local[2]. Reason: {primary_exc}"
                )
                conf.setMaster("local[2]")
                session = SparkSession.builder.config(conf=conf).getOrCreate()

            # Suppress verbose JVM / Spark logging
            cls._apply_log_level(session, resolved_log_level)

            cls._spark = session
            logger.info(
                f"SparkSession '{resolved_app_name}' successfully initialized "
                f"(Spark {session.version}, Master: {session.sparkContext.master})"
            )
            return cls._spark

    @staticmethod
    def _apply_log_level(session: SparkSession, log_level: str) -> None:
        """Suppress noisy Spark internal logs via SparkContext and JVM loggers."""
        try:
            sc = session.sparkContext
            sc.setLogLevel(log_level)

            # Access Log4j configuration if available in the JVM gateway
            jvm = sc._jvm
            if jvm is not None:
                # Log4j 2.x Support (Spark 3.3+)
                try:
                    configurator = jvm.org.apache.logging.log4j.core.config.Configurator
                    level = getattr(jvm.org.apache.logging.log4j.Level, log_level.upper())
                    configurator.setRootLevel(level)
                    configurator.setLevel("org.apache.spark", level)
                    configurator.setLevel("org.sparkproject", level)
                    configurator.setLevel("org.apache.hadoop", level)
                except Exception:
                    # Log4j 1.x Fallback
                    try:
                        logger_cls = jvm.org.apache.log4j.Logger
                        level_cls = jvm.org.apache.log4j.Level
                        target_lvl = level_cls.toLevel(log_level.upper())
                        logger_cls.getRootLogger().setLevel(target_lvl)
                        logger_cls.getLogger("org.apache.spark").setLevel(target_lvl)
                        logger_cls.getLogger("org.sparkproject").setLevel(target_lvl)
                        logger_cls.getLogger("org.apache.hadoop").setLevel(target_lvl)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Non-critical: Unable to fully set JVM log level: {e}")

    @classmethod
    def get_spark_info(cls) -> Dict[str, Any]:
        """Return diagnostic runtime metrics and metadata for the active session."""
        session = cls.get_active_session()
        if session is None:
            return {"status": "inactive"}

        sc = session.sparkContext
        return {
            "status": "active",
            "version": session.version,
            "app_name": sc.appName,
            "master": sc.master,
            "spark_user": sc.sparkUser(),
            "default_parallelism": sc.defaultParallelism,
            "driver_memory": sc.getConf().get("spark.driver.memory", "unknown"),
            "executor_memory": sc.getConf().get("spark.executor.memory", "unknown"),
            "shuffle_partitions": session.conf.get("spark.sql.shuffle.partitions", "unknown"),
        }

    @classmethod
    def stop(cls) -> None:
        """Gracefully release and terminate the active SparkSession."""
        with cls._lock:
            if cls._spark is not None:
                try:
                    logger.info("Stopping SentinelSpark SparkSession...")
                    cls._spark.stop()
                except Exception as e:
                    logger.warning(f"Exception during SparkSession stop: {e}")
                finally:
                    cls._spark = None

    # Context Manager Support
    def __enter__(self) -> SparkSession:
        return self.get_or_create()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


def get_spark_session(config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> SparkSession:
    """Convenience functional access to Singleton SparkSession."""
    return SparkSessionManager.get_or_create(config=config, **kwargs)


def stop_spark_session() -> None:
    """Convenience functional shutdown of Singleton SparkSession."""
    SparkSessionManager.stop()

