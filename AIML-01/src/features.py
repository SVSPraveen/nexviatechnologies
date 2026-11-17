"""features.py: Milestone 2 - Feature engineering for offline training and Feast feature store."""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def compute_engineered_features(input_csv: str, output_parquet: str) -> pd.DataFrame:
    print(f"[Milestone 2] Engineering features from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Base timestamp anchored to Jan 1, 2026
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    df["event_timestamp"] = df["step"].apply(lambda s: base_time + timedelta(hours=int(s)))
    df["created_timestamp"] = df["event_timestamp"]
    df["account_id"] = df["nameOrig"].astype(str)

    # Sort strictly by event_timestamp to avoid lookahead leakage
    df = df.sort_values(by=["account_id", "event_timestamp"]).reset_index(drop=True)

    print("Computing rolling transaction velocities and historical user averages...")
    
    # 1. Cumulative transaction count and rolling 1h/24h velocities per account
    # We use expanding window mean of amount up to previous transaction
    df["prev_amount_sum"] = df.groupby("account_id")["amount"].cumsum() - df["amount"]
    df["prev_tx_count"] = df.groupby("account_id").cumcount()
    
    # Amount versus user average (fallback to 1.0 if first transaction)
    user_avg = np.where(df["prev_tx_count"] > 0, df["prev_amount_sum"] / df["prev_tx_count"], df["amount"])
    df["amount_to_user_avg_ratio"] = np.round(df["amount"] / np.maximum(user_avg, 1.0), 4)

    # Rolling velocities using time windows
    velocities_1h = []
    velocities_24h = []

    # Fast group-wise rolling velocity calculation
    for _, group in df.groupby("account_id"):
        times = group["event_timestamp"].values
        n = len(times)
        for i in range(n):
            t_curr = times[i]
            # Past 1 hour (exclusive of future)
            t_1h = t_curr - np.timedelta64(1, "h")
            t_24h = t_curr - np.timedelta64(24, "h")
            
            c_1h = np.sum((times[:i] >= t_1h))
            c_24h = np.sum((times[:i] >= t_24h))
            velocities_1h.append(c_1h)
            velocities_24h.append(c_24h)

    df["velocity_1h"] = velocities_1h
    df["velocity_24h"] = velocities_24h

    # Device and location features
    df["device_changed_flag"] = df["device_changed"].astype(int)
    df["location_delta_km"] = df["location_delta_km"].astype(float)

    # One-hot encode transaction type for tabular model
    type_dummies = pd.get_dummies(df["type"], prefix="type", dtype=int)
    for col in ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        if col not in type_dummies.columns:
            type_dummies[col] = 0
            
    df = pd.concat([df, type_dummies], axis=1)

    # Select final Feast-ready schema
    feature_cols = [
        "account_id",
        "event_timestamp",
        "created_timestamp",
        "step",
        "amount",
        "velocity_1h",
        "velocity_24h",
        "amount_to_user_avg_ratio",
        "device_changed_flag",
        "location_delta_km",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
        "nameDest",
        "device_id",
        "card_id",
        "isFraud"
    ]
    final_df = df[feature_cols].copy()

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    final_df.to_parquet(output_parquet, index=False)
    print(f"[Milestone 2] Engineered dataset written to {output_parquet} ({len(final_df)} rows)")
    return final_df

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    in_csv = os.path.join(base_dir, "..", "data", "raw", "paysim_sample.csv")
    out_pq = os.path.join(base_dir, "..", "data", "processed", "features.parquet")
    compute_engineered_features(in_csv, out_pq)

