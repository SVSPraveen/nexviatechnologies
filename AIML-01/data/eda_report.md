# PaySim Exploratory Data Analysis & Class Imbalance Report

## 1. Class Distribution

- **Total Transactions:** 50,000
- **Legitimate Transactions (0):** 49,911 (99.822%)
- **Fraudulent Transactions (1):** 89 (0.178%)
- **Imbalance Ratio:** ~1:561

> **Key Takeaway:** The dataset displays severe class imbalance (strictly below 1% positive class). Accuracy is invalid as a performance metric; PR-AUC and Recall at high Precision must be used.

## 2. Transaction Type Distribution

| Transaction Type | Total Count | Fraud Count | Fraud Rate (%) |
| :--- | :--- | :--- | :--- |
| CASH_IN | 7,571 | 0 | 0.000% |
| CASH_OUT | 12,649 | 45 | 0.356% |
| DEBIT | 2,534 | 0 | 0.000% |
| PAYMENT | 17,357 | 0 | 0.000% |
| TRANSFER | 9,889 | 44 | 0.445% |

## 3. Financial Magnitude Analysis

- **Legitimate Mean Amount:** $452.83 (Median: $314.93)
- **Fraudulent Mean Amount:** $42,964.58 (Median: $30,187.46)
- **Multiplier:** Fraud amounts average **94.9x** higher than legitimate baseline activity.
