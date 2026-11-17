"""Deep & Cross Network (DCN) ranking model for candidate re-scoring.

Milestone 5: Train a deep ranking model (DCN or Wide & Deep) that re-scores the candidates.
Explicitly learns bounded-degree feature interactions via Cross layers alongside
deep non-linear representations via Deep MLP layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.config import RankerConfig

logger = logging.getLogger(__name__)


class CrossNetworkLayer(nn.Module):
    """Single layer of the Cross Network:
    x_{l+1} = x_0 * (x_l^T w_l) + b_l + x_l
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # xl^T w_l is a scalar per sample: B x 1
        scalar = torch.matmul(xl, self.weight)  # B x 1
        # x0 * scalar + bias + xl
        return x0 * scalar + self.bias + xl


class CrossNetwork(nn.Module):
    """Stack of L Cross Network layers."""

    def __init__(self, input_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([CrossNetworkLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class DeepCrossNetwork(nn.Module):
    """Deep & Cross Network (DCN) model for click/interaction probability ranking."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        genre_dim: int,
        config: Optional[RankerConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or RankerConfig()
        self.num_users = num_users
        self.num_items = num_items
        self.genre_dim = genre_dim
        emb_dim = self.config.embedding_dim

        # Categorical ID embeddings
        self.user_embedding = nn.Embedding(num_users + 1, emb_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)

        # Dense features: user_count, user_avg, item_pop, item_avg, user_item_genre_dot
        num_dense_features = 5
        self.total_input_dim = (emb_dim * 2) + num_dense_features

        # Cross Network
        self.cross_net = CrossNetwork(self.total_input_dim, num_layers=self.config.num_cross_layers)

        # Deep Network (MLP)
        layers: list[nn.Module] = []
        in_dim = self.total_input_dim
        for hidden_size in self.config.deep_layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))
            in_dim = hidden_size
        self.deep_net = nn.Sequential(*layers)

        # Combination layer: combines Cross Network output and Deep Network output
        final_dim = self.total_input_dim + in_dim
        self.prediction_head = nn.Linear(final_dim, 1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        # Mask out-of-range user and item IDs to 0 (designated unknown/padding)
        safe_user_ids = torch.where((user_ids > 0) & (user_ids <= self.num_users), user_ids, torch.zeros_like(user_ids))
        safe_item_ids = torch.where((item_ids > 0) & (item_ids <= self.num_items), item_ids, torch.zeros_like(item_ids))

        u_emb = self.user_embedding(safe_user_ids)
        i_emb = self.item_embedding(safe_item_ids)
        x0 = torch.cat([u_emb, i_emb, dense_features], dim=-1)

        x_cross = self.cross_net(x0)
        x_deep = self.deep_net(x0)

        x_comb = torch.cat([x_cross, x_deep], dim=-1)
        logits = self.prediction_head(x_comb).squeeze(-1)
        return torch.sigmoid(logits)


class BaselineRanker:
    """Baseline heuristic / popularity-weighted ranker for Milestone 7 A/B testing."""

    def __init__(self) -> None:
        pass

    def score_candidates(
        self,
        user_features: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[float]:
        """Compute baseline heuristic score using item popularity, rating, and genre overlap."""
        user_genre = user_features.get("user_genre_affinity", np.zeros(19))
        scores: list[float] = []
        for cand in candidates:
            pop = float(cand.get("item_popularity_score", 0.0))
            avg_r = float(cand.get("item_avg_rating", 3.0)) / 5.0
            item_genre = cand.get("item_genre_vector", np.zeros(19))
            genre_match = float(np.dot(user_genre, item_genre))
            # Heuristic composite score in [0, 1]
            score = 0.45 * pop + 0.35 * avg_r + 0.20 * min(1.0, genre_match)
            scores.append(score)
        return scores


class DCNRankerTrainer:
    """Trainer and scoring manager for Deep & Cross Network Ranker."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        genre_dim: int,
        config: Optional[RankerConfig] = None,
    ) -> None:
        self.config = config or RankerConfig()
        self.num_users = num_users
        self.num_items = num_items
        self.genre_dim = genre_dim
        self.model = DeepCrossNetwork(
            num_users=num_users,
            num_items=num_items,
            genre_dim=genre_dim,
            config=self.config,
        )

    def _extract_dense_features(
        self,
        user_feat: dict[str, Any],
        item_feat: dict[str, Any],
    ) -> list[float]:
        u_count = float(user_feat.get("user_interaction_count", 1.0)) / 100.0
        u_avg = float(user_feat.get("user_avg_rating", 3.0)) / 5.0
        i_pop = float(item_feat.get("item_popularity_score", 0.0))
        i_avg = float(item_feat.get("item_avg_rating", 3.0)) / 5.0
        
        u_g = user_feat.get("user_genre_affinity", np.zeros(self.genre_dim, dtype=np.float32))
        i_g = item_feat.get("item_genre_vector", np.zeros(self.genre_dim, dtype=np.float32))
        genre_match = float(np.dot(u_g, i_g))
        
        return [u_count, u_avg, i_pop, i_avg, genre_match]

    def train_epoch(
        self,
        interaction_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> float:
        """Train DCN for one epoch using Binary Cross-Entropy loss on click label."""
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_reg,
        )
        bce_loss = nn.BCELoss()

        user_map = user_features_df.set_index("userId").to_dict(orient="index")
        item_map = item_features_df.set_index("movieId").to_dict(orient="index")

        shuffled = interaction_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        batch_size = self.config.batch_size
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(shuffled), batch_size):
            batch = shuffled.iloc[i : i + batch_size]
            if len(batch) < 4:
                continue

            u_ids = []
            i_ids = []
            dense_list = []
            labels = []

            for _, row in batch.iterrows():
                uid = int(row["userId"])
                mid = int(row["movieId"])
                lbl = float(row["label"])

                u_f = user_map.get(uid, {})
                i_f = item_map.get(mid, {})

                u_ids.append(uid if 1 <= uid <= self.num_users else 0)
                i_ids.append(mid if 1 <= mid <= self.num_items else 0)
                dense_list.append(self._extract_dense_features(u_f, i_f))
                labels.append(lbl)

            t_u = torch.tensor(u_ids, dtype=torch.long)
            t_i = torch.tensor(i_ids, dtype=torch.long)
            t_dense = torch.tensor(dense_list, dtype=torch.float32)
            t_labels = torch.tensor(labels, dtype=torch.float32)

            optimizer.zero_grad()
            preds = self.model(t_u, t_i, t_dense)
            loss = bce_loss(preds, t_labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(1, num_batches)

    def fit(
        self,
        train_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> list[float]:
        """Train DCN ranking model across epochs."""
        losses = []
        for epoch in range(1, self.config.epochs + 1):
            loss = self.train_epoch(train_df, user_features_df, item_features_df)
            losses.append(loss)
            logger.info("DCN Ranker Epoch %d/%d - BCE Loss: %.4f", epoch, self.config.epochs, loss)
        return losses

    def rank_candidates(
        self,
        user_id: int,
        candidate_ids: list[int],
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> list[tuple[int, float]]:
        """Re-score candidates using DCN and return sorted (movieId, score) descending."""
        if not candidate_ids:
            return []

        self.model.eval()
        user_match = user_features_df[user_features_df["userId"] == user_id]
        u_f = user_match.iloc[0].to_dict() if not user_match.empty else {}

        item_map = item_features_df.set_index("movieId").to_dict(orient="index")

        safe_uid = user_id if 1 <= user_id <= self.num_users else 0
        u_ids = [safe_uid] * len(candidate_ids)
        i_ids = [cid if 1 <= cid <= self.num_items else 0 for cid in candidate_ids]
        dense_list = [self._extract_dense_features(u_f, item_map.get(cid, {})) for cid in candidate_ids]

        with torch.no_grad():
            t_u = torch.tensor(u_ids, dtype=torch.long)
            t_i = torch.tensor(i_ids, dtype=torch.long)
            t_dense = torch.tensor(dense_list, dtype=torch.float32)
            scores = self.model(t_u, t_i, t_dense).cpu().numpy()

        scored = [(int(cid), float(s)) for cid, s in zip(candidate_ids, scores)]
        # Sort descending by precision score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
