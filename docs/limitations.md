# SentinelSpark Operational Limitations & Production Scaling Guide

This document analyzes the operational boundaries, empirical limitations, and scaling recommendations for deploying **SentinelSpark** in real-world Security Operations Center (SOC) environments.

---

## 1. Real-World Machine Learning Considerations

### 1.1 Severe Class Imbalance in Rare Attack Taxonomies
In the UNSW-NB15 benchmark, representation across threat families varies by multiple orders of magnitude:
- High-volume classes like `Exploits` and `Generic` account for thousands of flows.
- Extreme minority classes like `Worms` (<0.1% representation) and `Backdoor` are underrepresented.

**Operational Impact:** Even with tree depth tuning and feature weighting, statistical learning models have fewer training exemplars to distinguish rare vectors from background noise, resulting in lower per-class F1-scores for ultra-rare threats.

**Mitigation Strategies:**
1. **SMOTE / Synthetic Oversampling:** Incorporate PySpark-compatible synthetic minority oversampling during Stage 2 feature engineering.
2. **Cost-Sensitive Loss Weighting:** Apply inverse class frequency weighting within `LogisticRegression` and tree estimators.
3. **Threshold Calibration:** Tune classification decision thresholds specifically for high-risk minority categories.

### 1.2 Encrypted Payloads & Statistical NetFlow Telemetry
With the widespread adoption of TLS 1.3 and HTTPS, deep packet inspection (DPI) of payload contents is often infeasible due to end-to-end encryption.
- SentinelSpark is intentionally designed around **flow telemetry** (duration, byte/packet volumes, timing jitters, connection dynamics) rather than raw payload strings.
- However, sophisticated attackers can shape traffic timing (e.g., adding artificial delays or padding packet sizes) to mimic benign browsing profiles.

### 1.3 Concept Drift & Zero-Day Polymorphism
Network intrusion signatures are non-stationary:
- Attack tooling updates rapidly; evasion techniques evolve to exploit newly discovered software vulnerabilities.
- Static models will inevitably experience **performance decay** as modern traffic deviates from the training distributions.
- **Production Requirement:** Implement automated performance monitoring on production telemetry and configure scheduled retraining triggers (e.g., bi-weekly or monthly cadence).

---

## 2. Single-Node (`local[*]`) vs. Distributed Cluster Execution

| Architectural Attribute | Single-Node (`local[*]`) | Distributed Cluster (YARN / Kubernetes) |
| :--- | :--- | :--- |
| **Executor Environment** | Local JVM threads inside host machine RAM | Distributed worker nodes managed by cluster orchestrator |
| **Optimal Shuffle Partitions** | $8 - 16$ partitions (prevents thread thrashing) | $200 - 2,000$ partitions (matches total cluster vCPUs) |
| **Memory Boundaries** | Constrained by local host RAM (e.g., 8GB - 32GB) | Horizontally scalable across hundreds of gigabytes of RAM |
| **Shuffle Spill to Disk** | Can cause heavy NVMe/SSD wear on deep grid searches | Distributed across high-throughput executor scratch disks |
| **Best Used For** | Prototyping, unit testing, CI/CD, local telemetry audits | Enterprise 24/7 stream scanning, terabyte-scale PCAP mining |

### Local Node Bottlenecks & JVM Heap Pressure
When running `python main.py train --tune` locally:
- PySpark `CrossValidator` with 3-fold cross-validation evaluates multiple estimator combinations simultaneously.
- Deep tree structures (`maxDepth > 10`) combined with large feature vectors can exhaust default JVM heap space, triggering extensive garbage collection (GC) pauses.
- **Configured Tuning:** SentinelSpark pre-tunes `config.yaml` with:
  ```yaml
  spark:
    driver_memory: "4g"
    executor_memory: "4g"
    sql:
      shuffle_partitions: 8
  ```

---

## 3. Production Scaling & Architecture Recommendations

To scale SentinelSpark from local batches to enterprise-grade continuous network protection:

```
[Network Taps / Zeek / Suricata]
               │
               ▼
    [Apache Kafka Stream]
               │
               ▼
[Spark Structured Streaming Driver]
               │
      ┌────────┴────────┐
      ▼                 ▼
[Stage 1 Workers] [Stage 1 Workers]
      │
      ├─────────────────┐
      ▼                 ▼
[Benign Flow Store] [Stage 2 Attack Profilers]
                        │
                        ▼
            [Elasticsearch / OpenSearch SOC SIEM]
```

### 3.1 Streaming Ingestion with Apache Kafka
Replace static CSV ingestion with **Spark Structured Streaming**:
```python
streaming_df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "kafka-broker:9092")
    .option("subscribe", "network-flow-telemetry")
    .load()
)
```
Trained `PipelineModel` checkpoints can be applied directly to streaming DataFrames with sub-second micro-batch latencies.

### 3.2 Cold Storage Tiering via Delta Lake / Parquet
- Store processed flows in columnar **Delta Lake** or **Parquet** partitioned by date and transport protocol: `data/processed/year=2026/month=10/proto=tcp/`.
- Enables rapid analytical slicing across billions of historical connection records using Spark SQL without full table scans.

### 3.3 Hardware Sizing Guidelines (Enterprise SOC)

| Sizing Tier | Daily Flow Volume | Spark Driver | Spark Workers | Total RAM |
| :--- | :--- | :--- | :--- | :--- |
| **Starter / Branch** | < 10M flows / day | 8 vCPU, 16GB RAM | 2 nodes (8 vCPU, 32GB each) | 80 GB |
| **Mid-Market Enterprise** | 10M - 100M flows / day | 16 vCPU, 32GB RAM | 6 nodes (16 vCPU, 64GB each) | 416 GB |
| **Tier-1 Telecom / Cloud** | > 1B flows / day | 32 vCPU, 64GB RAM | 20+ nodes (32 vCPU, 128GB each)| 2.6+ TB |

### 3.4 SIEM & SOAR Integration
- Forward high-confidence threat alerts (`prob > 0.90`) via Webhooks or Kafka topics directly to SIEM consoles (Splunk, Elastic SIEM, Microsoft Sentinel).
- Automate incident response actions (e.g., dynamically updating firewall rules to null-route offending external source IPs).
