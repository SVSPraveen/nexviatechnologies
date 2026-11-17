"""scoring_service.py: Milestone 5 - Real-time FastAPI fraud scoring consumer service.

Features:
- Sub-millisecond Feast online feature retrieval.
- XGBoost inference combined with PyG GNN fraud ring risk score.
- Decision policy engine (APPROVE, REVIEW, BLOCK).
- SQLite-backed alert ledger for live analyst triage and feedback.
"""

import os
import sys
import time
import json
import sqlite3
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from feast import FeatureStore

# Ensure local paths are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FEATURE_REPO_PATH = os.path.join(PROJECT_ROOT, "feature_repo")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "alerts.db")

# Pydantic Schemas
class TransactionPayload(BaseModel):
    transaction_id: Optional[str] = Field(default=None, description="Unique transaction ID")
    account_id: str = Field(..., description="Originating account identifier, e.g. C0000001")
    amount: float = Field(..., gt=0, description="Transaction monetary amount")
    type: str = Field(..., description="Transaction type: TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN")
    nameDest: Optional[str] = Field(default=None, description="Destination account identifier")
    device_id: Optional[str] = Field(default=None, description="Device fingerprint ID")
    card_id: Optional[str] = Field(default=None, description="Card ID")

class ScoreResponse(BaseModel):
    transaction_id: str
    account_id: str
    amount: float
    xgboost_score: float
    gnn_ring_score: float
    blended_fraud_score: float
    decision: str
    latency_ms: float
    features_used: Dict[str, Any]
    reasons: List[str]

class FeedbackPayload(BaseModel):
    alert_id: str
    disposition: str = Field(..., description="confirmed_fraud or false_positive")
    analyst_id: str
    notes: Optional[str] = None

# Global Model State Container
class ScoringEngine:
    def __init__(self):
        self.feature_store = None
        self.xgb_model = None
        self.gnn_scores = {}
        self.blend_config = {}
        self.feature_cols = []

    def load(self):
        print(f"Loading Feast online feature store from {FEATURE_REPO_PATH}...")
        self.feature_store = FeatureStore(repo_path=FEATURE_REPO_PATH)

        xgb_model_path = os.path.join(MODELS_PATH, "xgboost_model.json")
        xgb_meta_path = os.path.join(MODELS_PATH, "xgboost_model_metadata.json")
        print(f"Loading XGBoost model from {xgb_model_path}...")
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(xgb_model_path)

        with open(xgb_meta_path, "r") as f:
            meta = json.load(f)
            self.feature_cols = meta.get("feature_cols", [])

        blend_path = os.path.join(MODELS_PATH, "blend_config.json")
        with open(blend_path, "r") as f:
            self.blend_config = json.load(f)

        gnn_scores_path = os.path.join(MODELS_PATH, "gnn_entity_scores.json")
        if os.path.exists(gnn_scores_path):
            with open(gnn_scores_path, "r") as f:
                self.gnn_scores = json.load(f)
        else:
            self.gnn_scores = {}

        self.init_db()
        print("ScoringEngine loaded successfully.")

    def init_db(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    transaction_id TEXT,
                    account_id TEXT,
                    amount REAL,
                    type TEXT,
                    xgboost_score REAL,
                    gnn_ring_score REAL,
                    blended_score REAL,
                    decision TEXT,
                    status TEXT DEFAULT 'NEW',
                    feedback_label TEXT,
                    analyst_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def score(self, tx: TransactionPayload) -> ScoreResponse:
        t0 = time.perf_counter()
        tx_id = tx.transaction_id or f"tx_{int(time.time()*1000)}"

        # 1. Fetch online features from Feast
        entity_rows = [{"account_id": tx.account_id}]
        features_to_fetch = [
            "account_fraud_features:velocity_1h",
            "account_fraud_features:velocity_24h",
            "account_fraud_features:amount_to_user_avg_ratio",
            "account_fraud_features:device_changed_flag",
            "account_fraud_features:location_delta_km",
        ]

        try:
            feast_data = self.feature_store.get_online_features(
                features=features_to_fetch,
                entity_rows=entity_rows,
            ).to_dict()
            
            v1h = feast_data["velocity_1h"][0] or 0
            v24h = feast_data["velocity_24h"][0] or 0
            ratio = feast_data["amount_to_user_avg_ratio"][0] or 1.0
            dev_changed = feast_data["device_changed_flag"][0] or 0
            loc_delta = feast_data["location_delta_km"][0] or 0.0
        except Exception as e:
            # Graceful fallback to default cold-start values
            v1h, v24h, ratio, dev_changed, loc_delta = 0, 0, 1.0, 0, 0.0

        # Construct input dictionary for model
        input_dict = {
            "amount": tx.amount,
            "velocity_1h": v1h,
            "velocity_24h": v24h,
            "amount_to_user_avg_ratio": ratio,
            "device_changed_flag": dev_changed,
            "location_delta_km": loc_delta,
            "type_CASH_IN": 1 if tx.type == "CASH_IN" else 0,
            "type_CASH_OUT": 1 if tx.type == "CASH_OUT" else 0,
            "type_DEBIT": 1 if tx.type == "DEBIT" else 0,
            "type_PAYMENT": 1 if tx.type == "PAYMENT" else 0,
            "type_TRANSFER": 1 if tx.type == "TRANSFER" else 0,
        }

        # Align with model training feature order
        feature_vector = np.array([[input_dict.get(c, 0) for c in self.feature_cols]], dtype=np.float32)

        # 2. XGBoost Inference
        xgb_prob = float(self.xgb_model.predict_proba(feature_vector)[0, 1])

        # 3. GNN Fraud Ring Risk Lookup
        gnn_prob = float(self.gnn_scores.get(tx.account_id, 0.05))

        # 4. Ensemble Blend
        w_xgb = self.blend_config.get("xgboost_weight", 0.65)
        w_gnn = self.blend_config.get("gnn_weight", 0.35)
        blended_score = round(float(w_xgb * xgb_prob + w_gnn * gnn_prob), 4)

        # 5. Decision Policy
        thresh_block = self.blend_config.get("threshold_block", 0.90)
        thresh_review = self.blend_config.get("threshold_review", 0.70)

        if blended_score >= thresh_block:
            decision = "BLOCK"
        elif blended_score >= thresh_review:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        # 6. Reason Codes
        reasons = []
        if ratio > 5.0:
            reasons.append(f"AMOUNT_SPIKE_{ratio:.1f}X_USER_AVG")
        if v1h > 3:
            reasons.append("HIGH_VELOCITY_BURST")
        if dev_changed == 1:
            reasons.append("NEW_DEVICE_DETECTED")
        if gnn_prob > 0.5:
            reasons.append("FRAUD_RING_COMMUNITY_CLUSTER")

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Record to alerts table if flagged
        if decision in ["REVIEW", "BLOCK"]:
            self.persist_alert(
                tx_id=tx_id,
                account_id=tx.account_id,
                amount=tx.amount,
                tx_type=tx.type,
                xgb_score=xgb_prob,
                gnn_score=gnn_prob,
                blended_score=blended_score,
                decision=decision,
            )

        return ScoreResponse(
            transaction_id=tx_id,
            account_id=tx.account_id,
            amount=tx.amount,
            xgboost_score=round(xgb_prob, 4),
            gnn_ring_score=round(gnn_prob, 4),
            blended_fraud_score=blended_score,
            decision=decision,
            latency_ms=latency_ms,
            features_used=input_dict,
            reasons=reasons,
        )

    def persist_alert(self, tx_id: str, account_id: str, amount: float, tx_type: str,
                      xgb_score: float, gnn_score: float, blended_score: float, decision: str):
        alert_id = f"alt_{int(time.time()*1000)}_{account_id[-4:]}"
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO alerts (alert_id, transaction_id, account_id, amount, type,
                                        xgboost_score, gnn_ring_score, blended_score, decision)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (alert_id, tx_id, account_id, amount, tx_type, xgb_score, gnn_score, blended_score, decision))
                conn.commit()
        except Exception as e:
            print(f"Error persisting alert: {e}")

engine = ScoringEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield

app = FastAPI(
    title="Aegis Real-Time Fraud Scoring Service",
    version="1.0.0",
    description="Sub-millisecond fraud scoring engine combining Feast, XGBoost, and PyTorch Geometric GNN",
    lifespan=lifespan,
)

@app.get("/healthz")
def healthz():
    return {
        "status": "healthy",
        "service": "scoring-service",
        "models_loaded": engine.xgb_model is not None,
    }

@app.post("/v1/score", response_model=ScoreResponse)
def score_transaction(payload: TransactionPayload):
    return engine.score(payload)

@app.get("/v1/alerts")
def list_alerts(limit: int = 50):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

@app.post("/v1/feedback")
def submit_feedback(payload: FeedbackPayload):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            UPDATE alerts 
            SET status = 'RESOLVED', feedback_label = ?, analyst_id = ?
            WHERE alert_id = ?
        """, (payload.disposition, payload.analyst_id, payload.alert_id))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Alert ID not found")
    return {"status": "success", "alert_id": payload.alert_id, "disposition": payload.disposition}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scoring_service:app", host="0.0.0.0", port=8000, reload=False)

