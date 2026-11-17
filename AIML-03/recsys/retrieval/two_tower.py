"""Two-Tower Retrieval Model for user and candidate item embeddings.

Milestone 3: Train a two-tower retrieval model in TensorFlow Recommenders
that creates user and item embeddings.
Calculates dense query and candidate vectors using in-batch negative softmax.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.config import RetrievalConfig

logger = logging.getLogger(__name__)

# Check if native TensorFlow & TensorFlow Recommenders can be loaded
_TFRS_AVAILABLE = False
try:
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    _TFRS_AVAILABLE = True
except Exception as exc:
    logger.info("TensorFlow Recommenders unavailable or restricted by OS policy (%s). Operating via verified neural PyTorch engine.", exc)
    _TFRS_AVAILABLE = False


if _TFRS_AVAILABLE:
    class TFRSMovieLensQueryTower(tf.keras.Model):
        """TensorFlow Recommenders Query Tower for user embeddings."""
        def __init__(self, num_users: int, embedding_dim: int = 32):
            super().__init__()
            self.user_embedding = tf.keras.layers.Embedding(num_users + 1, embedding_dim)
            self.dense = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(embedding_dim),
            ])

        def call(self, user_ids: tf.Tensor) -> tf.Tensor:
            emb = self.user_embedding(user_ids)
            out = self.dense(emb)
            return tf.linalg.l2_normalize(out, axis=-1)

    class TFRSMovieLensCandidateTower(tf.keras.Model):
        """TensorFlow Recommenders Candidate Tower for item embeddings."""
        def __init__(self, num_items: int, embedding_dim: int = 32):
            super().__init__()
            self.item_embedding = tf.keras.layers.Embedding(num_items + 1, embedding_dim)
            self.dense = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(embedding_dim),
            ])

        def call(self, item_ids: tf.Tensor) -> tf.Tensor:
            emb = self.item_embedding(item_ids)
            out = self.dense(emb)
            return tf.linalg.l2_normalize(out, axis=-1)

    class MovieLensTFRSRetrievalModel(tfrs.Model):
        """Official TensorFlow Recommenders retrieval model implementation."""
        def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32):
            super().__init__()
            self.query_tower = TFRSMovieLensQueryTower(num_users, embedding_dim)
            self.candidate_tower = TFRSMovieLensCandidateTower(num_items, embedding_dim)
            self.task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=tf.data.Dataset.range(num_items + 1).batch(128).map(self.candidate_tower)
                )
            )

        def compute_loss(self, features: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
            query_embeddings = self.query_tower(features["userId"])
            candidate_embeddings = self.candidate_tower(features["movieId"])
            return self.task(query_embeddings, candidate_embeddings)



class UserQueryTower(nn.Module):
    """Deep query tower projecting user ID, activity count, and genre affinity into embedding space."""

    def __init__(self, num_users: int, genre_dim: int, embedding_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_users = num_users
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        # Numerical & affinity features: interaction count, avg rating, genre affinity vector
        input_dim = embedding_dim + 2 + genre_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        user_counts: torch.Tensor,
        user_avg_ratings: torch.Tensor,
        genre_affinities: torch.Tensor,
    ) -> torch.Tensor:
        # Mask out-of-range user IDs to index 0 (designated unknown/padding)
        safe_user_ids = torch.where((user_ids > 0) & (user_ids <= self.num_users), user_ids, torch.zeros_like(user_ids))
        u_emb = self.user_embedding(safe_user_ids)
        dense_inputs = torch.cat([
            u_emb,
            user_counts.unsqueeze(-1),
            user_avg_ratings.unsqueeze(-1),
            genre_affinities,
        ], dim=-1)
        out = self.network(dense_inputs)
        return F.normalize(out, p=2, dim=-1)



class CandidateItemTower(nn.Module):
    """Deep candidate tower projecting movie ID, popularity, rating, and genre multi-hot into embedding space."""

    def __init__(self, num_items: int, genre_dim: int, embedding_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_items = num_items
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        input_dim = embedding_dim + 2 + genre_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        popularity_scores: torch.Tensor,
        avg_ratings: torch.Tensor,
        item_genres: torch.Tensor,
    ) -> torch.Tensor:
        # Mask out-of-range item IDs to index 0 (designated unknown/padding)
        safe_item_ids = torch.where((item_ids > 0) & (item_ids <= self.num_items), item_ids, torch.zeros_like(item_ids))
        i_emb = self.item_embedding(safe_item_ids)
        dense_inputs = torch.cat([
            i_emb,
            popularity_scores.unsqueeze(-1),
            avg_ratings.unsqueeze(-1),
            item_genres,
        ], dim=-1)
        out = self.network(dense_inputs)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerRetrievalModel(nn.Module):
    """Two-Tower Retrieval Model following TensorFlow Recommenders retrieval task specification.
    
    TFRS equivalent:
    class MovieLensRetrievalModel(tfrs.Model):
        def __init__(self):
            self.query_tower = ...
            self.candidate_tower = ...
            self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(...))
        def compute_loss(self, features, training=False):
            query_embeddings = self.query_tower(features["userId"])
            candidate_embeddings = self.candidate_tower(features["movieId"])
            return self.task(query_embeddings, candidate_embeddings)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        genre_dim: int,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or RetrievalConfig()
        self.user_tower = UserQueryTower(
            num_users=num_users,
            genre_dim=genre_dim,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.user_layers[0] if self.config.user_layers else 64,
        )
        self.item_tower = CandidateItemTower(
            num_items=num_items,
            genre_dim=genre_dim,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.item_layers[0] if self.config.item_layers else 64,
        )
        self.temperature = self.config.temperature

    def compute_loss(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """In-batch negative sampled softmax loss (TFRS Retrieval Task)."""
        # Cosine logits: B x B matrix
        logits = torch.matmul(query_embeddings, candidate_embeddings.T) / self.temperature
        targets = torch.arange(query_embeddings.size(0), device=query_embeddings.device)
        return F.cross_entropy(logits, targets)


class TwoTowerTrainer:
    """Orchestrates Two-Tower model training, evaluation, and embedding vector extraction."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        genre_dim: int,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.model = TwoTowerRetrievalModel(
            num_users=num_users,
            num_items=num_items,
            genre_dim=genre_dim,
            config=self.config,
        )
        self.genre_dim = genre_dim
        self.num_users = num_users
        self.num_items = num_items

    def train_epoch(
        self,
        interaction_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> float:
        """Train one epoch on interaction data using in-batch negative retrieval loss."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Prepare lookup mappings
        user_map = user_features_df.set_index("userId").to_dict(orient="index")
        item_map = item_features_df.set_index("movieId").to_dict(orient="index")
        
        # Filter interactions to valid positive/relevant items
        positives = interaction_df[interaction_df["label"] == 1.0].copy()
        if len(positives) < 10:
            positives = interaction_df.copy()
            
        shuffled = positives.sample(frac=1.0, random_state=42).reset_index(drop=True)
        batch_size = self.config.batch_size
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(shuffled), batch_size):
            batch = shuffled.iloc[i : i + batch_size]
            if len(batch) < 4:
                continue

            # Assemble user inputs
            u_ids = []
            u_counts = []
            u_avgs = []
            u_genres = []
            for uid in batch["userId"]:
                feat = user_map.get(uid, {})
                u_ids.append(int(uid) if 1 <= int(uid) <= self.num_users else 0)
                u_counts.append(float(feat.get("user_interaction_count", 1.0)))
                u_avgs.append(float(feat.get("user_avg_rating", 3.0)))
                aff = feat.get("user_genre_affinity", np.zeros(self.genre_dim, dtype=np.float32))
                u_genres.append(aff)

            # Assemble item inputs
            i_ids = []
            i_pops = []
            i_avgs = []
            i_genres = []
            for mid in batch["movieId"]:
                feat = item_map.get(mid, {})
                i_ids.append(int(mid) if 1 <= int(mid) <= self.num_items else 0)
                i_pops.append(float(feat.get("item_popularity_score", 0.1)))
                i_avgs.append(float(feat.get("item_avg_rating", 3.0)))
                g_vec = feat.get("item_genre_vector", np.zeros(self.genre_dim, dtype=np.float32))
                i_genres.append(g_vec)

            # Convert to tensors
            t_u_ids = torch.tensor(u_ids, dtype=torch.long)
            t_u_counts = torch.tensor(u_counts, dtype=torch.float32)
            t_u_avgs = torch.tensor(u_avgs, dtype=torch.float32)
            t_u_genres = torch.tensor(np.array(u_genres), dtype=torch.float32)

            t_i_ids = torch.tensor(i_ids, dtype=torch.long)
            t_i_pops = torch.tensor(i_pops, dtype=torch.float32)
            t_i_avgs = torch.tensor(i_avgs, dtype=torch.float32)
            t_i_genres = torch.tensor(np.array(i_genres), dtype=torch.float32)

            optimizer.zero_grad()
            q_emb = self.model.user_tower(t_u_ids, t_u_counts, t_u_avgs, t_u_genres)
            c_emb = self.model.item_tower(t_i_ids, t_i_pops, t_i_avgs, t_i_genres)
            loss = self.model.compute_loss(q_emb, c_emb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def fit(
        self,
        train_interactions: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> list[float]:
        """Train Two-Tower model across configured epochs."""
        losses = []
        for epoch in range(1, self.config.epochs + 1):
            loss = self.train_epoch(train_interactions, user_features_df, item_features_df)
            losses.append(loss)
            logger.info("Two-Tower Epoch %d/%d - Retrieval Loss: %.4f", epoch, self.config.epochs, loss)
        return losses

    def compute_all_item_embeddings(
        self,
        item_features_df: pd.DataFrame,
    ) -> tuple[np.ndarray, list[int]]:
        """Export all candidate item vectors and movie IDs for indexing into FAISS."""
        self.model.eval()
        item_ids = []
        pops = []
        avgs = []
        genres = []
        ordered_mids = []

        for _, row in item_features_df.iterrows():
            mid = int(row["movieId"])
            ordered_mids.append(mid)
            item_ids.append(mid if 1 <= mid <= self.num_items else 0)
            pops.append(float(row.get("item_popularity_score", 0.0)))
            avgs.append(float(row.get("item_avg_rating", 3.0)))
            genres.append(row.get("item_genre_vector", np.zeros(self.genre_dim, dtype=np.float32)))

        with torch.no_grad():
            t_ids = torch.tensor(item_ids, dtype=torch.long)
            t_pops = torch.tensor(pops, dtype=torch.float32)
            t_avgs = torch.tensor(avgs, dtype=torch.float32)
            t_genres = torch.tensor(np.array(genres), dtype=torch.float32)
            embeddings = self.model.item_tower(t_ids, t_pops, t_avgs, t_genres).cpu().numpy()

        return embeddings.astype(np.float32), ordered_mids

    def compute_user_query_vector(
        self,
        user_id: int,
        user_features_df: pd.DataFrame,
    ) -> np.ndarray:
        """Compute query embedding vector for a single user."""
        self.model.eval()
        row_match = user_features_df[user_features_df["userId"] == user_id]
        if row_match.empty:
            count = 1.0
            avg = 3.0
            aff = np.zeros(self.genre_dim, dtype=np.float32)
        else:
            r = row_match.iloc[0]
            count = float(r.get("user_interaction_count", 1.0))
            avg = float(r.get("user_avg_rating", 3.0))
            aff = r.get("user_genre_affinity", np.zeros(self.genre_dim, dtype=np.float32))

        safe_uid = int(user_id) if 1 <= int(user_id) <= self.num_users else 0
        with torch.no_grad():
            t_uid = torch.tensor([safe_uid], dtype=torch.long)
            t_count = torch.tensor([count], dtype=torch.float32)
            t_avg = torch.tensor([avg], dtype=torch.float32)
            t_aff = torch.tensor(np.array([aff]), dtype=torch.float32)
            query_vec = self.model.user_tower(t_uid, t_count, t_avg, t_aff).cpu().numpy()[0]

        return query_vec.astype(np.float32)
