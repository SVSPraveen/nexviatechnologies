"""train_gnn.py: Milestone 4 - Graph Neural Network (PyTorch Geometric) for fraud ring detection and score blending."""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

class FraudRingGNN(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_channels: int = 64, out_channels: int = 1):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        out = self.conv2(h, edge_index)
        return out.squeeze(-1)

def build_graph_and_train(input_parquet: str, gnn_output_path: str, blend_output_path: str) -> dict:
    print(f"[Milestone 4] Building graph from {input_parquet}...")
    df = pd.read_parquet(input_parquet)

    # 1. Map all unique entities (Accounts, Devices, Cards) to integer IDs
    all_accounts = set(df["account_id"].unique()).union(set(df["nameDest"].unique()))
    all_devices = set(df["device_id"].dropna().unique())
    all_cards = set(df["card_id"].dropna().unique())

    node_to_id = {}
    id_to_node = {}
    for node in sorted(all_accounts):
        nid = len(node_to_id)
        node_to_id[node] = nid
        id_to_node[nid] = {"name": node, "type": "account"}
    
    for node in sorted(all_devices):
        nid = len(node_to_id)
        node_to_id[node] = nid
        id_to_node[nid] = {"name": node, "type": "device"}

    for node in sorted(all_cards):
        nid = len(node_to_id)
        node_to_id[node] = nid
        id_to_node[nid] = {"name": node, "type": "card"}

    n_nodes = len(node_to_id)
    print(f"Constructed heterogeneous entity graph: {n_nodes:,} total nodes "
          f"({len(all_accounts):,} accounts, {len(all_devices):,} devices, {len(all_cards):,} cards)")

    # 2. Build edges: Account -> Device, Account -> Card, Account -> Account
    src_list, dst_list = [], []

    for _, row in df.iterrows():
        orig_id = node_to_id[row["account_id"]]
        dest_id = node_to_id[row["nameDest"]]
        dev_id = node_to_id[row["device_id"]]
        card_id = node_to_id[row["card_id"]]

        # Account <-> Device
        src_list.extend([orig_id, dev_id])
        dst_list.extend([dev_id, orig_id])

        # Account <-> Card
        src_list.extend([orig_id, card_id])
        dst_list.extend([card_id, orig_id])

        # Account -> Account (payment / transfer)
        src_list.extend([orig_id, dest_id])
        dst_list.extend([dest_id, orig_id])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    print(f"Total graph edges (undirected): {edge_index.size(1):,}")

    # 3. Node features (degree, total amount, avg velocity, avg ratio)
    node_features = np.zeros((n_nodes, 4), dtype=np.float32)
    degrees = np.bincount(src_list, minlength=n_nodes)
    node_features[:, 0] = np.log1p(degrees)

    acc_agg = df.groupby("account_id").agg({
        "amount": ["sum", "count"],
        "velocity_24h": "mean",
        "amount_to_user_avg_ratio": "mean",
        "isFraud": "max",
    })

    node_labels = np.zeros(n_nodes, dtype=np.float32)
    account_mask = np.zeros(n_nodes, dtype=bool)

    for acc_name, row in acc_agg.iterrows():
        if acc_name in node_to_id:
            nid = node_to_id[acc_name]
            node_features[nid, 1] = np.log1p(float(row[("amount", "sum")]))
            node_features[nid, 2] = float(row[("velocity_24h", "mean")])
            node_features[nid, 3] = float(row[("amount_to_user_avg_ratio", "mean")])
            node_labels[nid] = float(row[("isFraud", "max")])
            account_mask[nid] = True

    x = torch.from_numpy(node_features)
    y = torch.from_numpy(node_labels)
    mask = torch.from_numpy(account_mask)

    # 4. Train GNN Model
    print("\nTraining PyTorch Geometric Fraud Ring GNN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running GNN training on device: {device}")

    model = FraudRingGNN(in_channels=4, hidden_channels=64, out_channels=1).to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    mask = mask.to(device)

    n_pos = int(y[mask].sum().item())
    n_neg = int((1.0 - y[mask]).sum().item())
    pos_weight = torch.tensor([float(n_neg / max(1, n_pos))], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    model.train()
    for epoch in range(1, 31):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[mask], y[mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 30:
            print(f"Epoch {epoch:02d}/30 | Loss: {loss.item():.4f}")

    # 5. Evaluate GNN Ring Detection
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        gnn_probs = torch.sigmoid(logits).cpu().numpy()

    y_eval = y[mask].cpu().numpy()
    p_eval = gnn_probs[account_mask]

    precision, recall, _ = precision_recall_curve(y_eval, p_eval)
    gnn_pr_auc = float(auc(recall, precision))
    gnn_roc_auc = float(roc_auc_score(y_eval, p_eval))

    print("=" * 60)
    print("GNN FRAUD RING DETECTION EVALUATION")
    print("=" * 60)
    print(f"GNN Precision-Recall AUC (PR-AUC): {gnn_pr_auc:.4f}")
    print(f"GNN ROC-AUC:                      {gnn_roc_auc:.4f}")
    print("=" * 60)

    # Save model weights
    os.makedirs(os.path.dirname(gnn_output_path), exist_ok=True)
    torch.save(model.state_dict(), gnn_output_path)
    print(f"Saved trained PyG GNN model to {gnn_output_path}")

    # Cache account-level GNN ring scores for real-time blend lookup
    account_scores = {}
    for acc_name in all_accounts:
        if acc_name in node_to_id:
            nid = node_to_id[acc_name]
            account_scores[acc_name] = round(float(gnn_probs[nid]), 4)

    scores_cache_path = os.path.join(os.path.dirname(gnn_output_path), "gnn_entity_scores.json")
    with open(scores_cache_path, "w") as f:
        json.dump(account_scores, f)
    print(f"Cached {len(account_scores):,} GNN account ring scores to {scores_cache_path}")

    # 6. Configure Ensemble Blend: XGBoost + GNN
    blend_config = {
        "xgboost_weight": 0.65,
        "gnn_weight": 0.35,
        "gnn_pr_auc": gnn_pr_auc,
        "gnn_roc_auc": gnn_roc_auc,
        "formula": "P_blend = 0.65 * P_xgboost + 0.35 * P_gnn",
        "description": "Monotonic weighted blend of tabular behavioral features and graph community ring risk",
        "threshold_review": 0.70,
        "threshold_block": 0.90,
    }

    os.makedirs(os.path.dirname(blend_output_path), exist_ok=True)
    with open(blend_output_path, "w") as f:
        json.dump(blend_config, f, indent=2)
    print(f"[Milestone 4] Blend configuration saved to {blend_output_path}")

    return blend_config

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_pq = os.path.join(base_dir, "..", "data", "processed", "features.parquet")
    gnn_model_path = os.path.join(base_dir, "..", "models", "gnn_model.pt")
    blend_path = os.path.join(base_dir, "..", "models", "blend_config.json")
    build_graph_and_train(data_pq, gnn_model_path, blend_path)

