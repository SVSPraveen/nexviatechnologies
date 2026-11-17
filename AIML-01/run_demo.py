"""run_demo.py: One-command end-to-end demonstration of the Fraud Detection Platform Prototype."""

import os
import sys
import time
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

def step(title: str):
    print("\n" + "=" * 70)
    print(f"▶ {title.upper()}")
    print("=" * 70)

def main():
    print("======================================================================")
    print("  AEGIS FRAUD INTELLIGENCE PLATFORM - END-TO-END PROTOTYPE DEMO")
    print("======================================================================")

    # 1. Milestone 1: EDA
    step("Milestone 1: Generating Dataset & Running EDA")
    gen_script = os.path.join(BASE_DIR, "src", "generate_dataset.py")
    eda_script = os.path.join(BASE_DIR, "src", "eda.py")
    subprocess.run([PYTHON_EXE, gen_script], check=True)
    subprocess.run([PYTHON_EXE, eda_script], check=True)

    # 2. Milestone 2: Feature Engineering & Feast
    step("Milestone 2: Engineering Features & Initializing Feast Feature Store")
    feat_script = os.path.join(BASE_DIR, "src", "features.py")
    init_feast_script = os.path.join(BASE_DIR, "src", "init_feature_store.py")
    subprocess.run([PYTHON_EXE, feat_script], check=True)
    subprocess.run([PYTHON_EXE, init_feast_script], check=True)

    # 3. Milestone 3: XGBoost Training & MLflow Tracking
    step("Milestone 3: Training XGBoost with Class Weights & Tracking in MLflow")
    xgb_script = os.path.join(BASE_DIR, "src", "train_xgboost.py")
    subprocess.run([PYTHON_EXE, xgb_script], check=True)

    # 4. Milestone 4: PyTorch Geometric GNN & Score Blending
    step("Milestone 4: Training PyTorch Geometric GNN for Fraud Rings & Score Blend")
    gnn_script = os.path.join(BASE_DIR, "src", "train_gnn.py")
    subprocess.run([PYTHON_EXE, gnn_script], check=True)

    # 5. Milestone 5: Real-Time Scoring & Streaming Test
    step("Milestone 5: Testing Real-Time Scoring Service & Ingesting Alerts")
    test_scoring_script = os.path.join(BASE_DIR, "tests", "test_scoring.py")
    subprocess.run([PYTHON_EXE, test_scoring_script], check=True)

    # 6. Milestone 6: Evidently AI Drift Detection & Automated Retraining
    step("Milestone 6: Running Evidently AI Drift Report & Retraining Trigger")
    drift_script = os.path.join(BASE_DIR, "src", "drift_monitor.py")
    subprocess.run([PYTHON_EXE, drift_script, "--simulate-drift"], check=True)

    # 7. Milestone 7: Integration Tests Passed
    step("Milestone 7: Verifying Integration Suite & Manifests")
    test_proto_script = os.path.join(BASE_DIR, "tests", "test_prototype.py")
    subprocess.run([PYTHON_EXE, test_proto_script], check=True)

    print("\n" + "=" * 70)
    print("  ALL 7 MILESTONES COMPLETED & VERIFIED SUCCESSFULLY!")
    print("======================================================================")
    print("\nTo launch the Analyst Dashboard:")
    print("  streamlit run src/dashboard.py")
    print("\nTo start the FastAPI Scoring Service:")
    print("  uvicorn src.scoring_service:app --port 8000")
    print("======================================================================\n")

if __name__ == "__main__":
    main()

