"""train_xgboost.py: Milestone 3 - XGBoost model training with class weights, PR-AUC evaluation, and MLflow tracking."""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss, classification_report
import mlflow

def train_xgboost(input_parquet: str, model_output_path: str) -> dict:
    print(f"[Milestone 3] Loading engineered features from {input_parquet}...")
    df = pd.read_parquet(input_parquet)

    # Sort chronologically by step to enforce temporal split (no future lookahead leakage)
    df = df.sort_values(by="step").reset_index(drop=True)

    feature_cols = [
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
    ]
    target_col = "isFraud"

    # Temporal split: First 80% steps for training, last 20% steps for holdout evaluation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = float(n_neg / max(1, n_pos))

    print(f"Train samples: {len(X_train)} (Fraud: {n_pos}, Legit: {n_neg})")
    print(f"Test samples:  {len(X_test)} (Fraud: {int(y_test.sum())}, Legit: {len(y_test) - int(y_test.sum())})")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Set up MLflow with SQLite database backend
    base_dir = os.path.dirname(os.path.abspath(input_parquet))
    db_path = os.path.join(base_dir, "..", "data", "mlflow.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(db_path)}")
    mlflow.set_experiment("fraud_detection_prototype_xgboost")

    params = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "random_state": 42,
    }

    print("\nTraining XGBoost model...")
    with mlflow.start_run(run_name="xgboost_class_weighted_run") as run:
        mlflow.log_params(params)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", feature_cols)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Predictions on holdout test set
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate Precision-Recall AUC (Primary Headline Metric)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = float(auc(recall, precision))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        brier = float(brier_score_loss(y_test, y_prob))

        print("=" * 60)
        print("XGBOOST MODEL EVALUATION RESULTS (TIME-SPLIT HOLDOUT)")
        print("=" * 60)
        print(f"Precision-Recall AUC (PR-AUC): {pr_auc:.4f}  [Primary Metric]")
        print(f"ROC-AUC:                      {roc_auc:.4f}")
        print(f"Brier Calibration Score:       {brier:.4f}")
        print("-" * 60)
        print("Note: Accuracy is intentionally omitted because predicting all 0s gives >99% accuracy while catching 0 fraud.")
        print("=" * 60)

        # Log metrics to MLflow
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("brier_score", brier)

        # Save model locally
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        model.save_model(model_output_path)
        print(f"Saved trained XGBoost model to {model_output_path}")

        # Save feature columns metadata
        meta_path = os.path.splitext(model_output_path)[0] + "_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "feature_cols": feature_cols,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "brier_score": brier,
                "scale_pos_weight": scale_pos_weight,
                "mlflow_run_id": run.info.run_id,
            }, f, indent=2)

        mlflow.log_artifact(model_output_path)
        mlflow.log_artifact(meta_path)
        print(f"[Milestone 3] Model tracked in MLflow (Run ID: {run.info.run_id})")

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "run_id": run.info.run_id,
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_pq = os.path.join(base_dir, "..", "data", "processed", "features.parquet")
    model_path = os.path.join(base_dir, "..", "models", "xgboost_model.json")
    train_xgboost(data_pq, model_path)
