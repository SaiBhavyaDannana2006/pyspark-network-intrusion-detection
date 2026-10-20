"""
PyTest Configuration & Shared Session Fixtures for SentinelSpark
"""

import pytest
from src.spark_session import SparkSessionManager


@pytest.fixture(scope="session")
def spark():
    """Shared single-instance SparkSession across the entire test suite."""
    session = SparkSessionManager.get_or_create(
        master="local[2]",
        app_name="SentinelSpark-PyTest-Session",
        driver_memory="2g",
        executor_memory="2g",
    )
    yield session
    SparkSessionManager.stop()
