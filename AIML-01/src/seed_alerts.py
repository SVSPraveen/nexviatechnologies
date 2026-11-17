"""seed_alerts.py: Seeds initial fraud and review alerts into data/alerts.db."""

import os
import sys
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from scoring_service import engine, TransactionPayload

def seed():
    engine.load()
    data_path = os.path.join(base_dir, "..", "data", "processed", "features.parquet")
    df = pd.read_parquet(data_path)
    
    fraud_sample = df[df["isFraud"] == 1].head(12)
    legit_sample = df[df["isFraud"] == 0].head(8)
    combined = pd.concat([fraud_sample, legit_sample]).sample(frac=1.0, random_state=42)

    count = 0
    for _, row in combined.iterrows():
        tx_type = "TRANSFER" if row.get("type_TRANSFER") == 1 else "CASH_OUT" if row.get("type_CASH_OUT") == 1 else "PAYMENT"
        tx = TransactionPayload(
            account_id=str(row["account_id"]),
            amount=float(row["amount"]),
            type=tx_type,
            device_id=str(row.get("device_id", "DEV_0001")),
            card_id=str(row.get("card_id", "CARD_0001")),
        )
        res = engine.score(tx)
        if res.decision in ["REVIEW", "BLOCK"]:
            count += 1

    print(f"Seeded {count} alerts into alerts database.")

if __name__ == "__main__":
    seed()

