"""drift_monitor.py: Milestone 6 - Evidently AI data drift monitoring and automated retraining trigger."""

import os
import sys
import json
import subprocess
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features.parquet")
HTML_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "drift_report.html")
METRICS_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "drift_metrics.json")
AUDIT_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "retrain_audit.json")

DRIFT_THRESHOLD_SHARE = 0.15  # Trigger retraining if >15% of features exhibit statistical drift

def run_drift_analysis(simulate_drift: bool = False) -> dict:
    print(f"[Milestone 6] Loading baseline feature dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    feature_cols = [
        "amount",
        "velocity_1h",
        "velocity_24h",
        "amount_to_user_avg_ratio",
        "device_changed_flag",
        "location_delta_km",
    ]

    # Split: first 70% as reference baseline, remaining 30% as current scoring window
    split_idx = int(len(df) * 0.70)
    reference_data = df.iloc[:split_idx][feature_cols].copy()
    current_data = df.iloc[split_idx:][feature_cols].copy()

    if simulate_drift:
        print("Simulating adversarial concept/covariate drift (injecting 5x velocity spike)...")
        current_data["velocity_1h"] = current_data["velocity_1h"] * 5 + 4
        current_data["amount"] = current_data["amount"] * 3.5

    print("Generating Evidently AI statistical data drift report...")
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)

    # Save HTML interactive report for analysts
    os.makedirs(os.path.dirname(HTML_REPORT_PATH), exist_ok=True)
    snapshot.save_html(HTML_REPORT_PATH)
    print(f"Evidently AI HTML drift report saved to {HTML_REPORT_PATH}")

    # Extract JSON metrics from snapshot
    summary_dict = snapshot.dict()
    metrics_list = summary_dict.get("metrics", [])
    
    drift_share = 0.0
    n_drifted = 0
    drift_details = {}

    for m in metrics_list:
        if isinstance(m, dict):
            val = m.get("value")
            cfg = m.get("config", {})
            if isinstance(val, dict) and "share" in val:
                drift_share = float(val["share"])
                n_drifted = int(val["count"])
            elif cfg.get("type") == "evidently:metric_v2:ValueDrift":
                col_name = cfg.get("column")
                p_val = float(val) if val is not None else 1.0
                drift_details[col_name] = {"p_value": p_val, "is_drifted": p_val < 0.05}

    total_cols = len(feature_cols)
    retrain_required = bool(drift_share >= DRIFT_THRESHOLD_SHARE)

    drift_results = {
        "dataset_drift_detected": retrain_required,
        "drift_share": round(drift_share, 4),
        "number_of_drifted_features": n_drifted,
        "total_monitored_features": total_cols,
        "drift_threshold_share": DRIFT_THRESHOLD_SHARE,
        "feature_details": drift_details,
        "retrain_triggered": retrain_required,
    }

    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(drift_results, f, indent=2)

    print("=" * 60)
    print("EVIDENTLY AI DATA DRIFT REPORT SUMMARY")
    print("=" * 60)
    print(f"Monitored Features:     {total_cols}")
    print(f"Drifted Features:       {n_drifted} ({drift_share * 100:.1f}%)")
    print(f"Retraining Required:    {retrain_required} (Threshold: {DRIFT_THRESHOLD_SHARE*100:.1f}%)")
    print("=" * 60)

    # Automated Retraining Trigger
    if retrain_required:
        trigger_automatic_retraining(drift_results)

    return drift_results

def trigger_automatic_retraining(drift_info: dict):
    print("\n[AUTOMATIC RETRAINING TRIGGERED] Drift threshold exceeded!")
    print("Executing automated retraining pipeline for XGBoost and PyG GNN...")

    python_exe = sys.executable

    # 1. Retrain XGBoost
    train_xgb_script = os.path.join(BASE_DIR, "train_xgboost.py")
    subprocess.run([python_exe, train_xgb_script], check=True)

    # 2. Retrain GNN
    train_gnn_script = os.path.join(BASE_DIR, "train_gnn.py")
    subprocess.run([python_exe, train_gnn_script], check=True)

    audit_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "reason": "Evidently AI Feature Drift Threshold Breached",
        "drift_metrics": drift_info,
        "status": "RETRAIN_SUCCESSFUL",
    }

    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(audit_entry, f, indent=2)

    print(f"[Milestone 6] Automated retraining complete. Audit record logged to {AUDIT_LOG_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-drift", action="store_true", help="Simulate drifted traffic to verify trigger")
    args = parser.parse_args()

    run_drift_analysis(simulate_drift=args.simulate_drift)
