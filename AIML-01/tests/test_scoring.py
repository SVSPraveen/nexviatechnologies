"""test_scoring.py: Unit test for ScoringEngine in Milestone 5."""

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, "..", "src")
sys.path.insert(0, src_dir)

from scoring_service import engine, TransactionPayload

def test_scoring():
    print("Testing ScoringEngine initialization...")
    engine.load()
    
    # 1. Test legitimate account & transaction
    tx_legit = TransactionPayload(
        account_id="C0000001",
        amount=45.50,
        type="PAYMENT",
    )
    res_legit = engine.score(tx_legit)
    print(f"Legit Transaction Score: {res_legit.blended_fraud_score:.4f} (XGB: {res_legit.xgboost_score:.4f}, GNN: {res_legit.gnn_ring_score:.4f}) | Decision: {res_legit.decision} | Latency: {res_legit.latency_ms}ms")
    assert res_legit.decision == "APPROVE"

    # 2. Test high-risk syndicate ring account
    tx_fraud = TransactionPayload(
        account_id="C0000004",
        amount=75000.0,
        type="TRANSFER",
    )
    res_fraud = engine.score(tx_fraud)
    print(f"Syndicate Ring Transaction Score: {res_fraud.blended_fraud_score:.4f} (XGB: {res_fraud.xgboost_score:.4f}, GNN: {res_fraud.gnn_ring_score:.4f}) | Decision: {res_fraud.decision} | Latency: {res_fraud.latency_ms}ms | Reasons: {res_fraud.reasons}")
    
    assert res_fraud.gnn_ring_score >= 0.80
    assert res_fraud.blended_fraud_score >= 0.50
    assert "FRAUD_RING_COMMUNITY_CLUSTER" in res_fraud.reasons
    print("ScoringEngine test PASSED successfully!")

if __name__ == "__main__":
    test_scoring()
