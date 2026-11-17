"""dashboard.py: Milestone 7 - Analyst Workbench Dashboard (Streamlit).

Features:
- Live alert triage queue with blended fraud scores and decision badges.
- Case inspection showing XGBoost score, GNN fraud ring risk, and Feast features.
- Human-in-the-loop analyst feedback submission (Confirmed Fraud vs False Positive).
- Evidently AI drift monitor and retraining audit visualizer.
"""

import os
import json
import sqlite3
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "alerts.db")
DRIFT_METRICS_PATH = os.path.join(PROJECT_ROOT, "data", "drift_metrics.json")
RETRAIN_AUDIT_PATH = os.path.join(PROJECT_ROOT, "data", "retrain_audit.json")

# Streamlit Page Configuration
st.set_page_config(
    page_title="Aegis Fraud Intelligence - Analyst Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Aegis Real-Time Fraud Intelligence Dashboard")
st.caption("Live Transaction Triage, Graph Ring Analysis, and Human-in-the-Loop Feedback Governance")

# Ensure alerts DB exists
def get_alerts_df():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM alerts ORDER BY created_at DESC", conn)
    return df

# Top Operational Metrics Bar
col1, col2, col3, col4 = st.columns(4)

alerts_df = get_alerts_df()
total_alerts = len(alerts_df)
pending_review = len(alerts_df[alerts_df["status"] == "NEW"]) if not alerts_df.empty else 0
blocked_count = len(alerts_df[alerts_df["decision"] == "BLOCK"]) if not alerts_df.empty else 0

drift_data = {}
if os.path.exists(DRIFT_METRICS_PATH):
    with open(DRIFT_METRICS_PATH, "r") as f:
        drift_data = json.load(f)

with col1:
    st.metric("Total Alerts Generated", total_alerts)
with col2:
    st.metric("Pending Analyst Triage", pending_review, delta=f"{pending_review} unassigned", delta_color="inverse")
with col3:
    st.metric("Critical Blocked", blocked_count, delta="High Risk", delta_color="inverse")
with col4:
    drift_share = drift_data.get("drift_share", 0.0) * 100.0
    drift_status = "DRIFT DETECTED" if drift_data.get("retrain_triggered", False) else "HEALTHY"
    st.metric("Evidently AI Drift", f"{drift_share:.1f}%", delta=drift_status, delta_color="normal" if drift_status == "HEALTHY" else "inverse")

st.divider()

# Tab Layout
tab_queue, tab_case, tab_drift = st.tabs([
    "📋 Live Alert Queue",
    "🔍 Case Review & Feedback",
    "📊 Drift Monitoring & Retraining"
])

# TAB 1: Alert Queue
with tab_queue:
    st.subheader("Incoming Fraud Alerts & Case Queue")
    
    if alerts_df.empty:
        st.info("No alerts generated yet. Run the Kafka producer or scoring test to populate incoming transactions.")
        if st.button("Generate Simulated Alert"):
            from scoring_service import engine, TransactionPayload
            engine.load()
            sample_tx = TransactionPayload(
                account_id="C0000004",
                amount=48500.0,
                type="TRANSFER",
            )
            engine.score(sample_tx)
            st.rerun()
    else:
        # Filters
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            status_filter = st.selectbox("Filter by Status", ["ALL", "NEW", "RESOLVED"])
        with fcol2:
            decision_filter = st.selectbox("Filter by Decision", ["ALL", "BLOCK", "REVIEW"])

        filtered_df = alerts_df.copy()
        if status_filter != "ALL":
            filtered_df = filtered_df[filtered_df["status"] == status_filter]
        if decision_filter != "ALL":
            filtered_df = filtered_df[filtered_df["decision"] == decision_filter]

        # Formatting table display
        display_cols = [
            "alert_id", "transaction_id", "account_id", "amount", "type",
            "xgboost_score", "gnn_ring_score", "blended_score", "decision", "status", "feedback_label", "created_at"
        ]
        st.dataframe(
            filtered_df[display_cols].style.format({
                "amount": "${:,.2f}",
                "xgboost_score": "{:.4f}",
                "gnn_ring_score": "{:.4f}",
                "blended_score": "{:.4f}",
            }),
            use_container_width=True,
            height=380,
        )

# TAB 2: Case Review & Feedback Disposition Form
with tab_case:
    st.subheader("Analyst Case Investigation & Ground Truth Feedback")
    if alerts_df.empty:
        st.info("No cases available for review.")
    else:
        alert_ids = alerts_df["alert_id"].tolist()
        selected_alert_id = st.selectbox("Select Case / Alert to Investigate:", alert_ids)
        
        case_data = alerts_df[alerts_df["alert_id"] == selected_alert_id].iloc[0]

        ccol1, ccol2 = st.columns([2, 1])
        with ccol1:
            st.markdown(f"### Case: `{case_data['alert_id']}`")
            st.markdown(f"**Transaction ID:** `{case_data['transaction_id']}` | **Account:** `{case_data['account_id']}`")
            st.markdown(f"**Amount:** `${case_data['amount']:,.2f}` | **Type:** `{case_data['type']}`")
            st.markdown(f"**Current Status:** `{case_data['status']}` | **Current Disposition:** `{case_data['feedback_label'] or 'UNLABELLED'}`")

            # Scores Breakdown
            st.markdown("#### Dual Model Scoring Attribution")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Tabular XGBoost", f"{case_data['xgboost_score']:.4f}")
            sc2.metric("PyG GNN Ring Risk", f"{case_data['gnn_ring_score']:.4f}")
            sc3.metric("Final Blended Score", f"{case_data['blended_score']:.4f}", delta=case_data["decision"], delta_color="inverse")

            st.progress(float(case_data["blended_score"]))

        with ccol2:
            st.markdown("### Record Analyst Ground Truth")
            st.caption("Submitting ground truth labels automatically feeds back into the mature retraining dataset.")
            
            analyst_name = st.text_input("Analyst Handle", value="analyst_sarah_c")
            notes = st.text_area("Investigative Notes / SAR Justification", placeholder="Document why this transaction is fraud or false positive...")
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🚨 Confirm Fraud", type="primary", use_container_width=True):
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("""
                            UPDATE alerts SET status = 'RESOLVED', feedback_label = 'CONFIRMED_FRAUD', analyst_id = ?
                            WHERE alert_id = ?
                        """, (analyst_name, selected_alert_id))
                        conn.commit()
                    st.success("Case resolved as CONFIRMED FRAUD and queued for model retraining.")
                    st.rerun()

            with btn_col2:
                if st.button("✅ False Positive", use_container_width=True):
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("""
                            UPDATE alerts SET status = 'RESOLVED', feedback_label = 'FALSE_POSITIVE', analyst_id = ?
                            WHERE alert_id = ?
                        """, (analyst_name, selected_alert_id))
                        conn.commit()
                    st.success("Case resolved as FALSE POSITIVE.")
                    st.rerun()

# TAB 3: Evidently AI Drift Monitoring & Retraining
with tab_drift:
    st.subheader("Evidently AI Statistical Drift Governance")
    
    if os.path.exists(DRIFT_METRICS_PATH):
        with open(DRIFT_METRICS_PATH, "r") as f:
            d_info = json.load(f)
            
        m1, m2, m3 = st.columns(3)
        m1.metric("Monitored Feature Columns", d_info.get("total_monitored_features", 6))
        m2.metric("Drifted Features Count", d_info.get("number_of_drifted_features", 0))
        m3.metric("Drift Threshold Cutoff", f"{d_info.get('drift_threshold_share', 0.15)*100:.1f}%")

        st.json(d_info)

    if os.path.exists(RETRAIN_AUDIT_PATH):
        st.markdown("### Retraining Audit Trail")
        with open(RETRAIN_AUDIT_PATH, "r") as f:
            st.json(json.load(f))
    else:
        st.info("No automatic retraining triggers recorded yet.")

