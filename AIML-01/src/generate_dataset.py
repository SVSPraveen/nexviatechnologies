"""generate_dataset.py: Generates a high-fidelity 50,000-row PaySim dataset with <1% class imbalance."""

import os
import random
import numpy as np
import pandas as pd

def generate_paysim_sample(output_path: str, n_rows: int = 50000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    print(f"Generating realistic PaySim transaction sample ({n_rows} rows)...")

    # Generate pool of accounts and devices
    n_users = 12000
    n_merchants = 2000
    users = [f"C{i:07d}" for i in range(1, n_users + 1)]
    merchants = [f"M{i:07d}" for i in range(1, n_merchants + 1)]
    devices = [f"DEV_{i:05d}" for i in range(1, 4000)]
    cards = [f"CARD_{i:05d}" for i in range(1, 6000)]

    # Fraud syndicate clusters (fraud rings)
    syndicate_users = users[:150]
    syndicate_devices = devices[:30]
    syndicate_cards = cards[:40]

    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    type_probs = [0.35, 0.20, 0.25, 0.05, 0.15]

    records = []
    
    # 30 days = 720 hours
    for i in range(n_rows):
        step = int((i / n_rows) * 720) + 1
        tx_type = np.random.choice(types, p=type_probs)
        
        # Determine if this row is part of a fraud ring / attack
        # Total fraud rate target: ~0.35% (under 1%)
        is_fraud = 0
        if random.random() < 0.0035 and tx_type in ["TRANSFER", "CASH_OUT"]:
            is_fraud = 1
            orig = random.choice(syndicate_users)
            dest = random.choice(syndicate_users)
            dev = random.choice(syndicate_devices)
            crd = random.choice(syndicate_cards)
            amount = round(float(np.random.exponential(scale=45000.0) + 5000.0), 2)
            device_changed = 1
            location_delta = round(float(np.random.uniform(250.0, 1200.0)), 1)
        else:
            orig = random.choice(users)
            dest = random.choice(merchants if tx_type == "PAYMENT" else users)
            dev = random.choice(devices)
            crd = random.choice(cards)
            amount = round(float(np.random.exponential(scale=450.0) + 5.0), 2)
            device_changed = 1 if random.random() < 0.04 else 0
            location_delta = round(float(np.random.exponential(scale=12.0)), 1)

        old_bal_orig = round(float(np.random.uniform(amount * 0.8, amount * 5.0)), 2)
        new_bal_orig = max(0.0, round(old_bal_orig - amount, 2)) if tx_type in ["TRANSFER", "CASH_OUT", "PAYMENT"] else round(old_bal_orig + amount, 2)

        old_bal_dest = round(float(np.random.uniform(100.0, 50000.0)), 2)
        new_bal_dest = round(old_bal_dest + amount, 2)

        records.append({
            "step": step,
            "type": tx_type,
            "amount": amount,
            "nameOrig": orig,
            "oldbalanceOrg": old_bal_orig,
            "newbalanceOrig": new_bal_orig,
            "nameDest": dest,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
            "device_id": dev,
            "card_id": crd,
            "device_changed": device_changed,
            "location_delta_km": location_delta,
            "isFraud": is_fraud,
            "isFlaggedFraud": 1 if amount > 200000 and is_fraud else 0
        })

    df = pd.DataFrame(records)
    
    fraud_count = df["isFraud"].sum()
    fraud_pct = (fraud_count / len(df)) * 100
    print(f"Generated {len(df)} transactions | Fraud count: {fraud_count} ({fraud_pct:.3f}%)")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")
    return df

if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "paysim_sample.csv")
    generate_paysim_sample(out_file)

