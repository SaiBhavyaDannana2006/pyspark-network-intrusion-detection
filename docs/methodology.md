# SentinelSpark Modeling & Evaluation Methodology

This document outlines the theoretical and empirical methodology implemented in **SentinelSpark**, explaining model selection criteria, metric formulation, feature engineering rationale, and zero-leakage validation protocols.

---

## 1. Problem Formulation: Asymmetric Cyber Defense

In Network Intrusion Detection Systems (NIDS), classification errors carry asymmetric operational penalties:
- **False Negative (FN) — Type II Error (Catastrophic):** An active intrusion or APT (Advanced Persistent Threat) bypasses the security perimeter undetected, resulting in data exfiltration, ransomware encryption, or credential theft.
- **False Positive (FP) — Type I Error (Manageable Cost):** A legitimate user flow triggers an alert, resulting in brief SOC analyst triage or automated secondary inspection.

Standard machine learning objectives that maximize raw classification accuracy are fundamentally flawed for intrusion detection: a trivial model predicting "Normal" on a dataset with 95% benign traffic achieves 95% accuracy while missing 100% of cyber attacks.

### SentinelSpark Core Principle
SentinelSpark addresses this asymmetry by decoupling threat detection into a **hierarchical two-stage defense pipeline**:
1. **Stage 1 Binary Gatekeeper:** Optimized specifically for **Attack Recall (True Positive Rate)** to ensure zero incursions slip through.
2. **Stage 2 Multiclass Threat Profiler:** Activated exclusively on flagged traffic to assign granular threat intelligence classifications without wasting compute cycles on cleared normal flows.

---

## 2. Zero-Leakage Preprocessing Protocol

Data leakage occurs when information from test or validation sets contaminates the model training process, leading to optimistic performance estimates that degrade in production. SentinelSpark strictly enforces zero-leakage via three safeguards:

### 2.1 Deduplication Tracking
Duplicate flow records can skew evaluation splits if identical instances appear across both train and test partitions. SentinelSpark performs exact duplicate elimination prior to splitting and logs deduplication metrics (`preprocessor.duplicate_count`).

### 2.2 Training-Fitted Parameter Isolation
All statistical transformations are fitted exclusively on the 70% training split:
- **Missing Value Imputation:** Numeric columns with nulls are imputed using medians computed solely on the training partition (`pyspark.ml.feature.Imputer`).
- **Zero-Leakage IQR Soft-Capping:** Continuous features frequently contain extreme outliers caused by volumetric attacks (e.g. DoS floods). Rather than dropping rows (which would eliminate attack records), SentinelSpark computes 25th and 75th percentiles on the training partition and soft-caps values to $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$.
- **Parameter Auditing:** Learned medians and clipping boundaries are exported to `models/preprocessor_params.json` and reused during validation, test evaluation, and live inference.

### 2.3 Non-Predictive Column Dropping
Identifiers such as `id`, `index`, and raw record counters (`_c0`) are dropped before vector assembly to prevent models from learning spurious row-order artifacts.

---

## 3. Behavioral Network Feature Engineering

Static packet headers (e.g., protocol, port) are insufficient to detect polymorphic malware or modern zero-day incursions. SentinelSpark extracts interaction dynamics that capture underlying transmission behaviors:

| Engineered Feature | Mathematical Formulation | Security Significance |
| :--- | :--- | :--- |
| **`packet_rate`** | $\frac{spkts + dpkts}{dur + 10^{-6}}$ | Detects sudden volumetric surges characteristic of DoS and brute-force scans. |
| **`byte_rate`** | $\frac{sbytes + dbytes}{dur + 10^{-6}}$ | Measures aggregate bandwidth throughput to flag high-speed data exfiltration. |
| **`avg_bytes_per_packet`** | $\frac{sbytes + dbytes}{spkts + dpkts + 10^{-6}}$ | Identifies anomalous MTU sizing (e.g. tiny TCP keep-alives vs massive buffer overflow payloads). |
| **`sbyte_ratio`** | $\frac{sbytes}{sbytes + dbytes + 10^{-6}}$ | Distinguishes asymmetric client-server interactions (e.g., download vs upload exfiltration). |
| **`packet_ratio`** | $\frac{spkts}{dpkts + 10^{-6}}$ | Captures unanswered probing requests (e.g. SYN floods with 0 return packets). |
| **`tcp_handshake_ratio`** | $\frac{synack}{tcprtt + 10^{-6}}$ | Analyzes TCP establishment latency, revealing connection stalling or spoofed responder IPs. |

> **Numerical Stability Safeguard:** All denominators incorporate an epsilon scalar ($+10^{-6}$) to ensure safe division on microsecond zero-duration network flows without producing `NaN` or `Inf`.

---

## 4. Stage 1 Binary Intrusion Detection

Stage 1 runs a competitive tournament across three distinct Spark MLlib estimators:

1. **Logistic Regression (`LogisticRegression`):**
   - High-throughput linear decision boundary with L2 regularization (`regParam=0.01`).
   - Serves as the ultra-low-latency gatekeeper suitable for microsecond line-rate deployment.
2. **Random Forest Classifier (`RandomForestClassifier`):**
   - Non-linear ensemble of 50 decision trees (`maxDepth=10`, `subsamplingRate=0.8`).
   - Resilient against feature collinearity and robust to noisy network flow metrics.
3. **Gradient-Boosted Trees (`GBTClassifier`):**
   - Sequentially boosts weak decision trees (`maxIter=30`, `stepSize=0.1`).
   - Captures subtle interaction boundaries between rare threat signatures.

### Model Selection Scoring Function
Candidate models are ranked on validation partitions using a composite weighted selection score:

$$\text{Selection Score} = 0.60 \times \text{Attack Recall (TPR)} + 0.40 \times \text{F1-Score}$$

Prioritizing Attack Recall ensures that models minimizing False Negatives are selected even if they exhibit marginal False Positive trade-offs.

---

## 5. Stage 2 Multiclass Threat Taxonomy Profiling

Once an anomaly is flagged by Stage 1, it enters Stage 2 for granular intelligence profiling.

### 5.1 Training Corpus Isolation
Stage 2 models are trained **strictly on malicious records** (`label == 1`). Normal traffic is excluded, preventing the majority benign class from dominating decision tree splits.

### 5.2 Taxonomy Classes (UNSW-NB15)
Stage 2 categorizes threats into 9 distinct intrusion families:
1. **Exploits:** Known vulnerability attacks targeting software bugs or buffer overflows.
2. **DoS (Denial of Service):** Exhaustion of target bandwidth or computational resources.
3. **Fuzzers:** Automated generation of random or malformed inputs to induce system crashes.
4. **Reconnaissance:** Port scanning, ping sweeps, and network topology discovery.
5. **Analysis:** Web-oriented incursions (e.g., cross-site scripting, SQL injection, port probes).
6. **Backdoor:** Covert remote access mechanisms bypassing authentication controls.
7. **Shellcode:** Injection of small executable payloads designed to spawn command shells.
8. **Worms:** Self-propagating autonomous network malware spreading without user action.
9. **Generic:** Cryptographic or cryptographic-hash collision attacks.

---

## 6. Metric Selection Rationale

| Metric | Formula | Why Chosen for SentinelSpark |
| :--- | :---: | :--- |
| **Attack Recall (TPR)** | $\frac{TP}{TP + FN}$ | **Primary Operational Metric.** Measures the percentage of actual intrusions detected. |
| **Precision** | $\frac{TP}{TP + FP}$ | Measures analyst alert fidelity; prevents SOC team alert fatigue. |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic balance between threat detection and false alarm suppression. |
| **ROC-AUC** | $\int_0^1 \text{TPR}(t) \, dt$ | Discriminative ability across all possible classification probability thresholds. |
| **PR-AUC** | $\int_0^1 \text{Precision}(r) \, dr$ | **Critical for Imbalanced Data.** Unlike ROC-AUC, PR-AUC focuses on minority attack classes. |
| **False Negative Rate (FNR)** | $\frac{FN}{FN + TP}$ | Direct indicator of security blindspots; must approach 0.0. |
| **False Positive Rate (FPR)** | $\frac{FP}{FP + TN}$ | Quantifies the false alarm rate on benign traffic. |
