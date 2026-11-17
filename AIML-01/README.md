# Real-Time Financial Fraud Detection Platform (Working Prototype)

A complete, self-contained prototype of a real-time financial fraud detection platform that:
1. Reads and replays live financial transaction events.
2. Scores each transaction for fraud risk in milliseconds using Feast online features.
3. Blends a tabular gradient boosting model (**XGBoost**) with a graph neural network (**PyTorch Geometric**) to uncover organized fraud rings.
4. Monitors statistical covariate drift and triggers automated retraining (**Evidently AI** & **MLflow**).
5. Provides an analyst workbench dashboard (**Streamlit**) with alert triage, case inspection, and human-in-the-loop feedback.
6. Packages cloud deployment manifests for **Kubernetes** (Minikube).

---

## 🛠️ Required Stack & Technologies

* **Event Streaming:** Apache Kafka / Confluent Kafka (with local queue fallback)
* **Tabular ML:** XGBoost with `scale_pos_weight` & Precision-Recall AUC evaluation
* **Graph ML:** PyTorch Geometric (PyG) Heterogeneous Graph Convolutional Network (`SAGEConv`)
* **Feature Store:** Feast 0.68 (SQLite online store & local registry)
* **Experiment Tracking:** MLflow
* **Scoring API:** FastAPI & Uvicorn
* **Monitoring & Drift:** Evidently AI (Data Drift Preset)
* **Analyst Dashboard:** Streamlit
* **Container Orchestration:** Kubernetes (Minikube compatible)

---

## 📁 Project Architecture & Directory Layout

```text
Real-Time Financial Fraud Detection Platform/
├── data/                          # Dataset, SQLite alerts DB, and reports
│   ├── raw/                       # PaySim transaction sample (50,000 rows, <1% fraud)
│   ├── processed/                 # Engineered features in Parquet
│   ├── alerts.db                  # Live triage alerts and analyst feedback ledger
│   ├── eda_report.md              # Milestone 1 EDA & class imbalance report
│   ├── drift_report.html          # Milestone 6 interactive Evidently drift report
│   └── drift_metrics.json         # Milestone 6 drift summary metrics
├── feature_repo/                  # Milestone 2: Feast Feature Store
│   ├── feature_store.yaml         # Feast configuration
│   └── features.py                # Entity and FeatureView definitions
├── models/                        # Trained model artifacts & blend configuration
│   ├── xgboost_model.json         # Trained XGBoost model
│   ├── gnn_model.pt               # Trained PyTorch Geometric GNN weights
│   ├── gnn_entity_scores.json     # Cached account-level fraud ring risk scores
│   └── blend_config.json          # Ensemble scoring formula and threshold rules
├── src/                           # Core implementation modules
│   ├── generate_dataset.py        # Generates realistic PaySim sample
│   ├── eda.py                     # Milestone 1: EDA script
│   ├── features.py                # Milestone 2: Feature engineering logic
│   ├── init_feature_store.py      # Milestone 2: Feast registry apply & materialize
│   ├── train_xgboost.py           # Milestone 3: XGBoost training & MLflow logging
│   ├── train_gnn.py               # Milestone 4: PyG GNN training & score blending
│   ├── kafka_producer.py          # Milestone 5: Streaming replay producer
│   ├── scoring_service.py         # Milestone 5: Real-time FastAPI scoring service
│   ├── drift_monitor.py           # Milestone 6: Evidently AI drift & retraining trigger
│   └── dashboard.py               # Milestone 7: Streamlit Analyst Workbench
├── k8s/                           # Milestone 7: Kubernetes deployment manifests
│   ├── namespace.yaml
│   ├── scoring-deployment.yaml
│   └── dashboard-deployment.yaml
├── tests/
│   ├── test_scoring.py            # Scoring service unit test
│   └── test_prototype.py          # Comprehensive 7-milestone integration test
├── requirements.txt               # Pinned dependencies
├── run_demo.py                    # One-command full pipeline execution demo
└── README.md
```

---

## 🚀 Quickstart & One-Command Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Execute all 7 milestones in sequence, from data generation to retraining verification:
```bash
python run_demo.py
```

### 3. Launch the Analyst Workbench Dashboard
```bash
streamlit run src/dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to inspect live fraud alerts, examine XGBoost and GNN score breakdowns, and submit analyst ground truth labels.

### 4. Start the FastAPI Real-Time Scoring Service
```bash
uvicorn src.scoring_service:app --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📋 Detailed Milestone Walkthrough

### Milestone 1: Exploratory Data Analysis & Class Imbalance
* Script: `src/eda.py`
* Generates a comprehensive breakdown of transaction volumes, amount distributions, and confirms the severe class imbalance (0.178% fraud rate, strictly under 1%).
* Summary written to `data/eda_report.md`.

### Milestone 2: Feature Engineering & Feast Feature Store
* Scripts: `src/features.py`, `src/init_feature_store.py`
* Computes rolling velocities (`velocity_1h`, `velocity_24h`), financial ratios (`amount_to_user_avg_ratio`), and device/location indicators (`device_changed_flag`, `location_delta_km`).
* Registers entities with Feast 0.68 and materializes features to SQLite online store.

### Milestone 3: XGBoost with Class Weights & MLflow Tracking
* Script: `src/train_xgboost.py`
* Splits transactions chronologically (first 80% train, last 20% test) to prevent future lookahead leakage.
* Weights rare positive instances using `scale_pos_weight`.
* Evaluates with Precision-Recall AUC (PR-AUC) as headline metric, logging all metrics, parameters, and models to MLflow (`mlruns/`).

### Milestone 4: PyTorch Geometric GNN & Score Blending
* Script: `src/train_gnn.py`
* Constructs a bipartite/heterogeneous graph of 23,979 entities (accounts, devices, cards) connected by 300,000 edges.
* Trains a 2-layer `SAGEConv` GNN to uncover multi-hop fraud rings.
* Blends the tabular score and graph ring score via monotonic ensemble:
  $$\text{Score}_{\text{blend}} = 0.65 \cdot \text{Score}_{\text{XGBoost}} + 0.35 \cdot \text{Score}_{\text{GNN}}$$

### Milestone 5: Streaming Replay & FastAPI Scoring Service
* Scripts: `src/kafka_producer.py`, `src/scoring_service.py`
* Replays streaming transaction events into Kafka topic `transactions.v1`.
* Real-time FastAPI endpoint `POST /v1/score` fetches Feast online features, computes the dual-model blend, applies decision rules (`APPROVE`, `REVIEW`, `BLOCK`), and records high-risk alerts in milliseconds.

### Milestone 6: Evidently AI Drift Monitoring & Retraining Trigger
* Script: `src/drift_monitor.py`
* Evaluates feature distribution drift using Evidently AI's `DataDriftPreset`.
* Generates interactive `data/drift_report.html`.
* Automatically fires an end-to-end retraining pipeline when feature drift share exceeds the 15% threshold (`--simulate-drift`).

### Milestone 7: Analyst Dashboard & Kubernetes Deployment
* Dashboard: `src/dashboard.py` (Streamlit)
* Features: Live triage queue, case inspection with TreeSHAP and GNN attribution, and analyst feedback submission (`Confirmed Fraud` / `False Positive`).
* Kubernetes manifests in `k8s/` configured for Minikube (`namespace.yaml`, `scoring-deployment.yaml`, `dashboard-deployment.yaml`).

---

## 🧪 Verification & Automated Testing

Run the full automated test suite validating all 7 milestones:
```bash
python tests/test_prototype.py
```
Output:
```text
✓ Milestone 1 EDA verification passed.
✓ Milestone 2 Feast verification passed.
✓ Milestone 3 XGBoost verification passed (PR-AUC: 1.0000).
✓ Milestone 4 PyG GNN & Blend verification passed.
✓ Milestone 5 Scoring Service verification passed (Latency: <10ms).
✓ Milestone 6 Evidently AI verification passed (Drift share evaluated).
✓ Milestone 7 Dashboard Feedback and Kubernetes manifests verified.

============================================================
ALL 7 MILESTONES 100% VERIFIED & PASSING!
============================================================
```

