# SentinelSpark System Architecture & Distributed Dataflow

This document provides a comprehensive technical overview of the **SentinelSpark** distributed Network Intrusion Intelligence platform, detailing execution boundaries, Apache Spark MLlib dataflows, and component responsibilities.

---

## 1. System Topology & Distributed Dataflow Diagram

The following diagram illustrates the complete end-to-end dataflow through Apache Spark driver and executor nodes:

```
+-------------------------------------------------------------------------------------------------------+
|                                    RAW TELEMETRY INGESTION LAYER                                      |
|                                                                                                       |
|   Official UNSW-NB15 CSV Partitions              CLI / Env Override            Synthetic Micro-Demo   |
|   (UNSW_NB15_1..4 / train/test-set.csv)        (DATASET_PATH)                (10K-15K Flows)          |
|                    \                                  |                               /               |
|                     \                                 |                              /                |
|                      +--------------------------------+-----------------------------+                 |
|                                                       |                                               |
|                                                       v                                               |
|                                     DataLoader.load_or_generate()                                     |
|                                                       |                                               |
|                                                       v                                               |
|                                       normalize_schema() [45 Features]                                |
|                                   (Lowercase, Trim, Type Alignment)                                   |
+-------------------------------------------------------+-----------------------------------------------+
                                                        |
                                                        v
+-------------------------------------------------------+-----------------------------------------------+
|                                ZERO-LEAKAGE PREPROCESSING & SPLITTING                                 |
|                                                                                                       |
|                  Train Split (70%)             Validation Split (15%)          Test Split (15%)       |
|                         │                               │                             │               |
|                         v                               │                             │               |
|             fit_transform_train()                       │                             │               |
|             • Exact Dedup Tracking                      │                             │               |
|             • Spark Median Imputer ─────────────────────┼─────────────────────────────┤               |
|             • IQR Soft-Capping Bounds ──────────────────┼─────────────────────────────┤               |
|                         │                               │                             │               |
|                         │                               v                             v               |
|                         │                     transform_eval()              transform_eval()          |
|                         │                     (Fitted Medians)              (Fitted Medians)          |
|                         │                     (Fitted Bounds)               (Fitted Bounds)           |
|                         │                               │                             │               |
|                         v                               v                             v               |
|                  Clean Train DF                   Clean Val DF                  Clean Test DF         |
+-------------------------------------------------------+-----------------------------------------------+
                                                        |
                                                        v
+-------------------------------------------------------+-----------------------------------------------+
|                                    NETWORK BEHAVIORAL FEATURE ENGINEERING                             |
|                                                                                                       |
|   • Packet Rates: (spkts + dpkts) / (dur + 1e-6)        • Byte Density: sbytes / (spkts + 1e-6)       |
|   • Transmission Rates: (sbytes + dbytes)/(dur + 1e-6)  • Asymmetry: spkts / (dpkts + 1e-6)           |
|   • Source Byte Ratio: sbytes / (sbytes + dbytes+1e-6)  • TCP Dynamics: synack / (tcprtt + 1e-6)      |
|   • Target Leakage Drop: Exclude 'id', 'index', '_c0', non-predictive identifiers                     |
+-------------------------------------------------------+-----------------------------------------------+
                                                        |
                                                        v
+-------------------------------------------------------+-----------------------------------------------+
|                                      SPARK MLLIB FEATURE PIPELINE                                     |
|                                                                                                       |
|   Categorical Features (proto, service, state)   Numeric Features (dur, spkts, rates, loads, ratios)  |
|                         │                                                     │                       |
|                         v                                                     │                       |
|                   StringIndexer                                               │                       |
|                         │                                                     │                       |
|                         v                                                     │                       |
|                 OneHotEncoderEstimator                                        │                       |
|                         │                                                     │                       |
|                         +──────────────────────────┬──────────────────────────+                       |
|                                                    │                                                  |
|                                                    v                                                  |
|                                             VectorAssembler                                           |
|                                                    │                                                  |
|                                                    v                                                  |
|                                             StandardScaler                                            |
|                                                    │                                                  |
|                                                    v                                                  |
|                                        [Assembled Feature Vector]                                     |
+----------------------------------------------------+--------------------------------------------------+
                                                     │
                                                     v
+----------------------------------------------------+--------------------------------------------------+
|                                    TWO-STAGE ML CLASSIFIER ARCHITECTURE                               |
|                                                                                                       |
|    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐    |
|    │ STAGE 1: HIGH-THROUGHPUT BINARY GATEKEEPER                                                  │    |
|    │  Estimators: Logistic Regression, Random Forest, Gradient-Boosted Trees (GBT)               │    |
|    │  Objective Function: Maximize Attack Recall (TPR) & Harmonic F1-Score                       │    |
|    │  Decision Criterion: Clear normal traffic; forward anomalies to Stage 2                    │    |
|    └───────────────────────────────────────────────┬─────────────────────────────────────────────┘    |
|                                                    │                                                  |
|                     ┌──────────────────────────────┴──────────────────────────────┐                   |
|                     │                                                             │                   |
|                     v                                                             v                   |
|           Prediction = Normal (0)                                       Prediction = Attack (1)       |
|                     │                                                             │                   |
|                     v                                                             v                   |
|             [Cleared Benign Flow]                         ┌──────────────────────────────────────┐    |
|                                                           │ STAGE 2: MULTICLASS THREAT PROFILER  │    |
|                                                           │  Input: Attacks Only (label == 1)    │    |
|                                                           │  Estimator: Multiclass Random Forest │    |
|                                                           │  Labels: 9 Attack Categories         │    |
|                                                           └───────────────────┬──────────────────┘    |
|                                                                               │                       |
|                                                                               v                       |
|                                                                    [Threat Taxonomy Vector]           |
|                                                                     • Exploits     • DoS              |
|                                                                     • Fuzzers      • Reconnaissance   |
|                                                                     • Analysis     • Backdoor         |
|                                                                     • Shellcode    • Worms            |
|                                                                     • Generic                         |
+-------------------------------------------------------------------------------+-----------------------+
                                                                                │
                                                                                v
+-------------------------------------------------------------------------------+-----------------------+
|                                       SERVING & OPERATIONAL CONSOLE                                   |
|                                                                                                       |
|           Batch & CLI Driver               Inference Engine API             Streamlit SOC Dashboard   |
|               (main.py)                    (src/inference.py)                      (app.py)           |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Component Breakdown

### 2.1 SparkSession Factory (`src/spark_session.py`)
- **Singleton Lifecycle:** Enforces a single initialized `SparkSession` per Python process, preventing JVM deadlocks and multi-session metastore collisions.
- **Kryo Serialization:** Utilizes `org.apache.spark.serializer.KryoSerializer` with 512MB buffer limits for low-latency vector serialization across Spark executors.
- **Dynamic Memory Ceilings:** Automatically provisions `spark.driver.memory` (default 4GB) and sets default shuffle partitions to align with workstation core counts (`local[*]`).

### 2.2 Data Ingestion & Sanitization (`src/data_loader.py`)
- **Discovery Strategy:** Hierarchical path resolution prioritizing CLI flags, `DATASET_PATH` environment variables, split pairs (`UNSW_NB15_training-set.csv` / `testing-set.csv`), partitioned chunks (`UNSW-NB15_1..4.csv`), and built-in synthetic generation.
- **Schema Canonicalization:** Strips leading/trailing spaces, converts headers to lowercase, detects missing headers (`_c0`, `_c1`), and casts all 45 columns to canonical PySpark types (`DoubleType`, `LongType`, `IntegerType`, `StringType`).

### 2.3 Zero-Leakage Preprocessing (`src/preprocessing.py`)
- **Deduplication:** Tracks and removes exact duplicate network flows before dataset partitioning.
- **Parameter Fitting Isolation:** PySpark `Imputer` and IQR outlier bounds are computed **strictly on the training split**.
- **Audit Serialization:** Imputed medians and outlier bounds are saved to `models/preprocessor_params.json` for validation and inference scoring.

### 2.4 Behavioral Feature Engineering (`src/feature_engineering.py`)
- **Rate & Ratio Derivation:** Constructs safe numerical features capturing transport asymmetries (`packet_rate`, `byte_rate`, `sbyte_ratio`, `tcp_handshake_ratio`, `bytes_per_packet_src`).
- **Target Leakage Shielding:** Explicitly drops non-predictive columns (`id`, `index`, `_c0`) and isolates the Stage 2 training corpus from benign records (`label == 0`).

### 2.5 MLlib Feature Pipeline (`src/feature_pipeline.py`)
- Encapsulates PySpark transformers into an atomic `Pipeline`:
  1. `StringIndexer`: Encodes categoricals (`proto`, `service`, `state`) into numerical indexes with `handleInvalid="keep"`.
  2. `OneHotEncoder`: Expands indexed categoricals into sparse binary indicator vectors.
  3. `VectorAssembler`: Concatenates continuous numerical features and one-hot vectors into a single feature vector (`features`).
  4. `StandardScaler`: Normalizes vector elements to zero mean and unit variance (`withStd=True`, `withMean=False`).

### 2.6 Dual-Stage Detection & Profiling
- **Stage 1 Binary Tournament (`src/binary_detection.py`):** Trains Logistic Regression, Random Forest, and Gradient-Boosted Trees. Model selection optimizes a weighted score prioritizing **Attack Recall (60%)** over **F1-Score (40%)**.
- **Stage 2 Multiclass Profiler (`src/attack_profiling.py`):** Trains a Multiclass Random Forest on detected attack instances to classify specific threat families.

### 2.7 Model Persistence & Real-Time Inference (`src/inference.py`)
- **PipelineModel Export:** Uses native Spark ML persistence (`PipelineModel.save()`) to preserve end-to-end preprocessing, transformations, and trained estimator weights.
- **InferenceEngine:** Provides high-throughput scoring on Pandas DataFrames, CSV paths, and single network flow telemetry records.

---

## 3. Distributed Execution & Scalability Considerations

1. **Shuffle Partition Tuning:** Default PySpark shuffle partitions (200) are optimized down to `8` in `local[*]` mode to prevent thread thrashing and memory overhead on developer machines.
2. **Columnar In-Memory Caching:** Training and validation DataFrames are explicitly persisted in memory (`df.persist(StorageLevel.MEMORY_AND_DISK)`) across tournament iterations and unpersisted upon training completion.
3. **Partition Parallelism:** When scaled to a cluster (YARN/Kubernetes/Standalone), data partitions map directly to executor cores, allowing horizontal scaling across terabyte-scale PCAP telemetry logs.
