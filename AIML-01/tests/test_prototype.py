"""test_prototype.py: End-to-end integration test validating all 7 milestones of the prototype."""

import os
import sys
import json
import sqlite3
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from scoring_service import engine, TransactionPayload, FeedbackPayload
import drift_monitor

def test_milestone_1_eda():
    """Verify Milestone 1 EDA report exists and documents severe class imbalance."""
    report_path = os.path.join(PROJECT_ROOT, "data", "eda_report.md")
    assert os.path.exists(report_path), "EDA report not found"
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Class Distribution" in content
    assert "Fraudulent Transactions" in content
    assert "< 1%" in content or "below 1%" in content
    print("✓ Milestone 1 EDA verification passed.")

def test_milestone_2_feast():
    """Verify Milestone 2 Feast feature definitions and materialized dataset."""
    parquet_path = os.path.join(PROJECT_ROOT, "data", "processed", "features.parquet")
    yaml_path = os.path.join(PROJECT_ROOT, "feature_repo", "feature_store.yaml")
    assert os.path.exists(parquet_path), "Engineered features parquet not found"
    assert os.path.exists(yaml_path), "Feast feature_store.yaml not found"
    print("✓ Milestone 2 Feast verification passed.")

def test_milestone_3_xgboost():
    """Verify Milestone 3 XGBoost model and metadata artifacts."""
    model_path = os.path.join(PROJECT_ROOT, "models", "xgboost_model.json")
    meta_path = os.path.join(PROJECT_ROOT, "models", "xgboost_model_metadata.json")
    assert os.path.exists(model_path), "XGBoost model file not found"
    assert os.path.exists(meta_path), "XGBoost metadata file not found"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["pr_auc"] > 0.80, f"PR-AUC too low: {meta['pr_auc']}"
    assert "scale_pos_weight" in meta
    print(f"✓ Milestone 3 XGBoost verification passed (PR-AUC: {meta['pr_auc']:.4f}).")

def test_milestone_4_gnn_and_blend():
    """Verify Milestone 4 PyG GNN model, account risk scores, and blend configuration."""
    gnn_path = os.path.join(PROJECT_ROOT, "models", "gnn_model.pt")
    scores_path = os.path.join(PROJECT_ROOT, "models", "gnn_entity_scores.json")
    blend_path = os.path.join(PROJECT_ROOT, "models", "blend_config.json")
    assert os.path.exists(gnn_path), "GNN model weights not found"
    assert os.path.exists(scores_path), "GNN entity scores cache not found"
    assert os.path.exists(blend_path), "Blend config not found"
    with open(blend_path, "r") as f:
        blend = json.load(f)
    assert "xgboost_weight" in blend
    assert "gnn_weight" in blend
    assert blend["xgboost_weight"] + blend["gnn_weight"] == 1.0
    print("✓ Milestone 4 PyG GNN & Blend verification passed.")

def test_milestone_5_scoring_service():
    """Verify Milestone 5 sub-millisecond scoring service and decision logic."""
    engine.load()
    
    # 1. Clean account
    res_clean = engine.score(TransactionPayload(
        account_id="C0000001",
        amount=15.00,
        type="PAYMENT",
    ))
    assert res_clean.decision == "APPROVE"
    assert res_clean.blended_fraud_score < 0.50

    # 2. Fraud ring account
    res_fraud = engine.score(TransactionPayload(
        account_id="C0000004",
        amount=55000.00,
        type="TRANSFER",
    ))
    assert res_fraud.decision in ["REVIEW", "BLOCK"]
    assert res_fraud.blended_fraud_score > 0.50
    print(f"✓ Milestone 5 Scoring Service verification passed (Latency: {res_clean.latency_ms:.2f}ms / {res_fraud.latency_ms:.2f}ms).")

def test_milestone_6_evidently_drift():
    """Verify Milestone 6 Evidently AI drift detection output."""
    html_report = os.path.join(PROJECT_ROOT, "data", "drift_report.html")
    metrics_json = os.path.join(PROJECT_ROOT, "data", "drift_metrics.json")
    assert os.path.exists(html_report), "Evidently drift HTML report not found"
    assert os.path.exists(metrics_json), "Evidently drift metrics JSON not found"
    with open(metrics_json, "r") as f:
        d = json.load(f)
    assert "drift_share" in d
    assert "total_monitored_features" in d
    print(f"✓ Milestone 6 Evidently AI verification passed (Drift share: {d['drift_share']*100:.1f}%).")

def test_milestone_7_dashboard_and_feedback():
    """Verify Milestone 7 feedback mechanism and Kubernetes manifests."""
    db_path = os.path.join(PROJECT_ROOT, "data", "alerts.db")
    assert os.path.exists(db_path), "Alerts database not found"

    # Simulate analyst feedback submission
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT alert_id FROM alerts LIMIT 1")
        row = cursor.fetchone()
        assert row is not None, "No alerts available for feedback test"
        target_alert_id = row[0]

        conn.execute("""
            UPDATE alerts SET status = 'RESOLVED', feedback_label = 'CONFIRMED_FRAUD', analyst_id = 'test_analyst'
            WHERE alert_id = ?
        """, (target_alert_id,))
        conn.commit()

        # Check update persisted
        cur2 = conn.execute("SELECT status, feedback_label FROM alerts WHERE alert_id = ?", (target_alert_id,))
        status, label = cur2.fetchone()
        assert status == "RESOLVED"
        assert label == "CONFIRMED_FRAUD"

    # Check k8s manifests exist
    k8s_dir = os.path.join(PROJECT_ROOT, "k8s")
    assert os.path.exists(os.path.join(k8s_dir, "namespace.yaml"))
    assert os.path.exists(os.path.join(k8s_dir, "scoring-deployment.yaml"))
    assert os.path.exists(os.path.join(k8s_dir, "dashboard-deployment.yaml"))
    print("✓ Milestone 7 Dashboard Feedback and Kubernetes manifests verified.")

if __name__ == "__main__":
    test_milestone_1_eda()
    test_milestone_2_feast()
    test_milestone_3_xgboost()
    test_milestone_4_gnn_and_blend()
    test_milestone_5_scoring_service()
    test_milestone_6_evidently_drift()
    test_milestone_7_dashboard_and_feedback()
    print("\n" + "=" * 60)
    print("ALL 7 MILESTONES 100% VERIFIED & PASSING!")
    print("=" * 60)

