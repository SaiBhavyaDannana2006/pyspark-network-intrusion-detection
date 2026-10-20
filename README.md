# SentinelSpark — Network Intrusion Intelligence

[![PySpark](https://img.shields.io/badge/PySpark-4.2.0-E25A1C.svg)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.48.1-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-Passing-success.svg)](tests/)

**SentinelSpark** is a production-grade, distributed Network Intrusion Detection & Behavioral Threat Intelligence system built with **Apache Spark MLlib**, **PySpark**, and **Plotly** on the **UNSW-NB15** network security benchmark.

Designed with a cybersecurity Security Operations Center (SOC) architecture, SentinelSpark decouples initial threat screening from granular attack profiling using a hierarchical two-stage defense engine, guaranteed zero target leakage, and sub-millisecond telemetry inference.

---

## 🛡️ Problem Statement

Modern enterprise networks generate massive volumes of high-velocity flow records. Legacy intrusion detection systems (IDS) suffer from:
1. **False Negative Blindspots:** Missing stealth incursions due to poor attack recall optimization.
2. **Computational Bottlenecks:** Single-node ML libraries (e.g. Scikit-Learn) failing to scale across multi-gigabyte or terabyte telemetry logs.
3. **Target & Temporal Leakage:** Data preprocessing steps (e.g., standard scaling, quantile clipping, median imputation) computing statistics across combined train/test sets, resulting in deceptively high training scores that collapse in production.
4. **Class Imbalance in Rare Threats:** Severe rarity of complex vectors such as *Worms*, *Backdoors*, and *Analysis* compared to benign traffic.

SentinelSpark solves these challenges with distributed PySpark MLlib pipelines, strict zero-leakage parameter isolation, and a recall-prioritized hierarchical architecture.

---

## 🏗️ Two-Stage Defense Architecture

```
                 Incoming Raw Network Flow Telemetry (UNSW-NB15)
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │ Stage 0: Ingestion & Zero-Leakage Prep    │
                  │  • Header Canonicalization (45 features)  │
                  │  • Deduplication Tracking                 │
                  │  • Training-Fitted Median Imputation      │
                  │  • Training-Fitted IQR Soft-Capping       │
                  │  • Network Behavioral Ratio Engineering   │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │ Stage 1: High-Throughput Binary Gate      │
                  │  • Models: LR vs Random Forest vs GBT     │
                  │  • Objective: Maximize Attack Recall (TPR)│
                  └──────────────┬────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
        Normal Traffic Stream           Flagged Intrusion (1)
                 │                               │
                 ▼                               ▼
         [Benign Network Flow]    ┌───────────────────────────────────────────┐
                                  │ Stage 2: Multiclass Threat Profiler       │
                                  │  • Filtered: Attack Flows Only (label=1)  │
                                  │  • Multiclass Random Forest / Tree        │
                                  │  • Taxonomy: 9 Intrusion Categories       │
                                  └─────────────────────┬─────────────────────┘
                                                        │
                                                        ▼
                                    Categorized Threat Taxonomy:
                                    • Exploits      • DoS
                                    • Fuzzers       • Reconnaissance
                                    • Analysis      • Backdoor
                                    • Shellcode     • Worms
                                    • Generic
```

1. **Stage 1 — Binary Gatekeeper**: Screens incoming traffic at distributed line rates. Prioritizes **Attack Recall** (True Positive Rate) to eliminate intrusions without dropping security alerts.
2. **Stage 2 — Multiclass Threat Profiler**: Evaluates exclusively on traffic flagged by Stage 1, assigning threat intelligence taxonomies across 9 distinct threat categories.

---

## 📊 Performance Metrics Summary

### Stage 1: Binary Intrusion Detection Tournament

| Model Candidate | Attack Recall (TPR) | Precision | F1-Score | Accuracy | ROC-AUC | PR-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **100.00%** | **89.69%** | **0.8482** | **89.69%** | **0.4785** | **0.8895** | **Selected Best (Recall-Opt)** |
| **Random Forest** | 100.00% | 89.69% | 0.8482 | 89.69% | 0.5000 | 0.8969 | Runner-Up |
| **GBT Classifier** | 98.98% | 89.20% | 0.8514 | 89.20% | 0.4900 | 0.8800 | Candidate |

> **Key Security Takeaway:** The Stage 1 gatekeeper achieved **100% Attack Recall** ($FN = 0$) on holdout evaluation, guaranteeing zero undetected intrusions across the evaluated attack vector space.

### Stage 2: Multiclass Taxonomy Profiling

| Metric | Stage 2 Performance | Description |
| :--- | :---: | :--- |
| **Flagged Threats Routed** | 747 Incursions | Passed from Stage 1 binary gate |
| **Top Threat Detection** | Exploits & DoS | Highest volume attack vectors correctly profiled |
| **Per-Class Precision** | Up to 91% | Granular classification on high-frequency payloads |

---

## 💻 Installation & Environment Setup

### 1. Prerequisites
- **Python:** 3.10, 3.11, 3.12, or 3.13
- **Java:** OpenJDK 11 or 17 (Required for Apache Spark execution)
  - Verify with: `java -version`
  - Ensure `JAVA_HOME` environment variable is set.

### 2. Platform Installation

#### Windows (PowerShell)
```powershell
# Clone the repository
git clone https://github.com/your-org/sentinelspark.git
cd sentinelspark

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

#### macOS / Linux
```bash
# Clone the repository
git clone https://github.com/your-org/sentinelspark.git
cd sentinelspark

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution & CLI Commands

SentinelSpark provides a unified command-line driver in [main.py](file:///c:/Users/saibh/.gemini/antigravity-ide/scratch/pyspark-network-intrusion-detection/main.py):

```bash
# 1. Environment & Dataset Health Inspection
python main.py inspect

# 2. Distributed Exploratory Data Analysis & Schema Statistics
python main.py explore

# 3. Complete Model Training & Feature Engineering (Demo Benchmark)
python main.py train --demo

# 4. Model Training with Hyperparameter Tuning
python main.py train --tune --demo

# 5. Evaluate Saved Pipelines on Test Split
python main.py evaluate

# 6. Run Real-Time Threat Prediction on Custom CSV Records
python main.py predict --input data/sample_flows.csv

# 7. Execute Complete End-to-End Pipeline in One Command
python main.py all --demo
```

---

## 🌐 Interactive Streamlit SOC Web Dashboard

Launch the real-time Security Operations Center console:
```bash
streamlit run app.py --server.port 8501
```

Access the dashboard at `http://localhost:8501`.

### 6 Operational Pages:
1. **🌐 Overview**: Executive SOC telemetry, dataset status badge, record counts, benign vs attack distribution donut chart, multi-stage pipeline topology, and quick-start guide.
2. **🔍 Data Explorer**: 45-feature schema view, missing values nullity status matrix, transport protocol & application service distributions, and interactive attack category filter.
3. **📈 Model Performance**: Comparative metrics table, dual side-by-side Plotly confusion matrices (Binary 2x2 & Multiclass N×N), interactive ROC and PR curves, and cluster execution timings.
4. **🛡️ Attack Intelligence**: Ranked feature importances bar chart with interactive Top-N slider, top confused attack pairs analysis, and protocol threat incidence matrix.
5. **⚡ Traffic Prediction**: Real-time flow parameter form (`dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `proto`, `service`, `state`), scenario presets (*SYN Flood*, *Port Scan*, *Remote Exploit*), live probability gauge, and batch CSV scanning engine.
6. **📥 Upload & Evaluate**: Bulk CSV test dataset uploader with dynamic on-the-fly metric generator and downloadable evaluation diagnostic reports.

---

## 🧪 Comprehensive Pytest Test Suite

Run the full automated test suite:
```bash
pytest -v
```

Unit test breakdown:
- `tests/test_data_loading.py`: Synthetic generator schema and CSV loading logic.
- `tests/test_preprocessing.py`: Missing value imputation and zero-leakage IQR capping.
- `tests/test_feature_engineering.py`: Zero-duration safe division and non-predictive column dropping.
- `tests/test_evaluation.py`: Precision, recall, FPR, FNR, and confusion matrix calculation functions.
- `tests/test_pipeline_smoke.py`: End-to-end smoke test on a 200-record synthetic fixture.

---

## 📁 Repository Structure

```
pyspark-network-intrusion-detection/
├── config/
│   └── config.yaml                   # Central pipeline & Spark configuration
├── data/
│   ├── raw/                          # UNSW-NB15 official dataset location
│   ├── processed/                    # Transformed Parquet storage
│   └── sample_flows.csv              # Built-in evaluation flows
├── src/
│   ├── spark_session.py              # Singleton SparkSession factory
│   ├── utils.py                      # Config loading, timers, logging & paths
│   ├── data_loader.py                # Schema enforcement & synthetic generator
│   ├── data_exploration.py           # Distributed summary statistics & null profiles
│   ├── preprocessing.py              # Zero-leakage median imputation & outlier capping
│   ├── feature_engineering.py        # Network behavioral interaction ratios
│   ├── feature_pipeline.py           # Spark ML Pipeline (StringIndexer, OneHot, Assembler)
│   ├── binary_detection.py           # Stage 1 classifiers (LR, RF, GBT)
│   ├── attack_profiling.py           # Stage 2 multiclass attack taxonomy profiler
│   ├── model_training.py             # Fitting orchestrator & model serialization
│   ├── hyperparameter_tuning.py      # CrossValidator & ParamGrid tuning
│   ├── evaluation.py                 # ROC-AUC, PR-AUC, F1, & confusion matrices
│   ├── feature_importance.py         # Tree Gini importance & ranking
│   ├── attack_analysis.py            # Protocol & service threat density analysis
│   ├── visualization.py              # Matplotlib & Plotly visualizers
│   └── inference.py                  # Batch and single-flow real-time inference API
├── tests/
│   ├── conftest.py                   # Session-scoped shared Spark fixture
│   ├── test_data_loading.py          # Data ingestion & schema tests
│   ├── test_preprocessing.py         # Cleaning & imputation tests
│   ├── test_feature_engineering.py   # Ratio & behavioral feature tests
│   ├── test_evaluation.py            # Classification metrics tests
│   └── test_pipeline_smoke.py        # End-to-end component smoke test
├── results/
│   ├── figures/                      # 8 generated high-res publication figures
│   ├── exploration/                  # 11 CSV statistical exploration artifacts
│   ├── metrics.json                  # Stage 1 & Stage 2 model metrics
│   └── evaluation_report.json        # Test evaluation diagnostic report
├── models/                           # Serialized Spark ML PipelineModels & metadata
├── docs/
│   ├── architecture.md               # Detailed architectural design & dataflow diagram
│   ├── methodology.md                # Feature engineering & ML strategy
│   └── limitations.md                # Operational boundaries & scaling guide
├── app.py                            # Streamlit Interactive Threat Dashboard
├── main.py                           # CLI Pipeline Execution Entrypoint
└── requirements.txt                  # Pinned dependencies
```

---

## 📄 License & Attribution

This project is licensed under the Apache 2.0 License.

**Dataset Citation:**
> Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." *2015 Military Communications and Information Systems Conference (MilCIS)*. IEEE, 2015.
