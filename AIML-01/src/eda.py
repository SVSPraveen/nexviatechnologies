"""eda.py: Milestone 1 - Exploratory Data Analysis & Class Imbalance Analysis for PaySim."""

import os
import pandas as pd
import numpy as np

def run_eda(input_csv: str, output_report: str) -> dict:
    print(f"[Milestone 1] Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    total_records = len(df)
    fraud_records = int(df["isFraud"].sum())
    legit_records = total_records - fraud_records
    fraud_rate_pct = (fraud_records / total_records) * 100.0

    print("=" * 60)
    print("PAY-SIM EXPLORATORY DATA ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Transactions: {total_records:,}")
    print(f"Legitimate Count:  {legit_records:,} ({100 - fraud_rate_pct:.3f}%)")
    print(f"Fraudulent Count:  {fraud_records:,} ({fraud_rate_pct:.3f}%)")
    print(f"Class Imbalance:   1 fraud per {total_records / max(1, fraud_records):.1f} transactions (Heavy Imbalance < 1%)")
    print("-" * 60)

    # Transaction type breakdown
    type_breakdown = df.groupby("type")["isFraud"].agg(["count", "sum"]).reset_index()
    type_breakdown.columns = ["type", "total_transactions", "fraud_count"]
    type_breakdown["fraud_rate_pct"] = (type_breakdown["fraud_count"] / type_breakdown["total_transactions"]) * 100.0
    print("\nTransaction Type Breakdown:")
    print(type_breakdown.to_string(index=False))

    # Amount statistics
    legit_amounts = df[df["isFraud"] == 0]["amount"]
    fraud_amounts = df[df["isFraud"] == 1]["amount"]

    stats = {
        "total_records": total_records,
        "fraud_records": fraud_records,
        "fraud_rate_pct": fraud_rate_pct,
        "legit_mean_amount": float(legit_amounts.mean()),
        "fraud_mean_amount": float(fraud_amounts.mean()),
        "legit_median_amount": float(legit_amounts.median()),
        "fraud_median_amount": float(fraud_amounts.median()),
    }

    # Generate Markdown Report
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("# PaySim Exploratory Data Analysis & Class Imbalance Report\n\n")
        f.write("## 1. Class Distribution\n\n")
        f.write(f"- **Total Transactions:** {total_records:,}\n")
        f.write(f"- **Legitimate Transactions (0):** {legit_records:,} ({(100 - fraud_rate_pct):.3f}%)\n")
        f.write(f"- **Fraudulent Transactions (1):** {fraud_records:,} ({fraud_rate_pct:.3f}%)\n")
        f.write(f"- **Imbalance Ratio:** ~1:{int(total_records / max(1, fraud_records))}\n\n")
        f.write("> **Key Takeaway:** The dataset displays severe class imbalance (strictly below 1% positive class). Accuracy is invalid as a performance metric; PR-AUC and Recall at high Precision must be used.\n\n")
        
        f.write("## 2. Transaction Type Distribution\n\n")
        f.write("| Transaction Type | Total Count | Fraud Count | Fraud Rate (%) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for _, row in type_breakdown.iterrows():
            f.write(f"| {row['type']} | {int(row['total_transactions']):,} | {int(row['fraud_count']):,} | {row['fraud_rate_pct']:.3f}% |\n")
        f.write("\n")

        f.write("## 3. Financial Magnitude Analysis\n\n")
        f.write(f"- **Legitimate Mean Amount:** ${stats['legit_mean_amount']:,.2f} (Median: ${stats['legit_median_amount']:,.2f})\n")
        f.write(f"- **Fraudulent Mean Amount:** ${stats['fraud_mean_amount']:,.2f} (Median: ${stats['fraud_median_amount']:,.2f})\n")
        f.write(f"- **Multiplier:** Fraud amounts average **{stats['fraud_mean_amount'] / max(1.0, stats['legit_mean_amount']):.1f}x** higher than legitimate baseline activity.\n")

    print(f"\n[Milestone 1] Successfully generated EDA report: {output_report}")
    return stats

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_csv = os.path.join(base_dir, "..", "data", "raw", "paysim_sample.csv")
    report_md = os.path.join(base_dir, "..", "data", "eda_report.md")
    run_eda(data_csv, report_md)

