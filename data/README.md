# UNSW-NB15 Dataset Instructions

This directory houses the raw and processed data partitions for SentinelSpark.

## Directory Structure
- `data/raw/`: Place the official UNSW-NB15 CSV files here:
  - `UNSW_NB15_training-set.csv` (82,332 records)
  - `UNSW_NB15_testing-set.csv` (175,341 records)
  - Or `UNSW-NB15_combined.csv`
- `data/processed/`: Generated Parquet partitions with engineered features.

## Automatic Discovery & Synthetic Fallback
SentinelSpark features automated dataset discovery:
1. If standard UNSW-NB15 CSV files exist in `data/raw/`, they are ingested with strict schema enforcement.
2. If files are not present and `allow_synthetic_fallback: true` is configured in `config/config.yaml`, the pipeline automatically synthesizes realistic UNSW-NB15 network flows to allow zero-configuration local runs and continuous integration tests.

## Dataset Reference
Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." *2015 Military Communications and Information Systems Conference (MilCIS)*. IEEE, 2015.
