"""Unit tests for Milestone 3: Two-Tower Retrieval Model and Embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from recsys.config import RetrievalConfig
from recsys.retrieval.two_tower import (
    CandidateItemTower,
    TwoTowerRetrievalModel,
    TwoTowerTrainer,
    UserQueryTower,
)


def test_user_query_tower_forward_shape() -> None:
    """Verify query tower produces L2-normalized vectors of expected dimension."""
    tower = UserQueryTower(num_users=100, genre_dim=19, embedding_dim=16, hidden_dim=32)
    u_ids = torch.tensor([1, 2, 5], dtype=torch.long)
    u_counts = torch.tensor([10.0, 5.0, 1.0], dtype=torch.float32)
    u_avgs = torch.tensor([4.0, 3.5, 5.0], dtype=torch.float32)
    u_genres = torch.zeros((3, 19), dtype=torch.float32)

    emb = tower(u_ids, u_counts, u_avgs, u_genres)
    assert emb.shape == (3, 16)
    # Check L2 unit norm
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_candidate_item_tower_forward_shape() -> None:
    """Verify item tower produces L2-normalized vectors of expected dimension."""
    tower = CandidateItemTower(num_items=50, genre_dim=19, embedding_dim=16, hidden_dim=32)
    i_ids = torch.tensor([10, 20], dtype=torch.long)
    i_pops = torch.tensor([0.8, 0.2], dtype=torch.float32)
    i_avgs = torch.tensor([4.2, 3.0], dtype=torch.float32)
    i_genres = torch.zeros((2, 19), dtype=torch.float32)

    emb = tower(i_ids, i_pops, i_avgs, i_genres)
    assert emb.shape == (2, 16)
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_two_tower_retrieval_loss_computation() -> None:
    """Verify retrieval loss computes valid in-batch softmax cross-entropy."""
    model = TwoTowerRetrievalModel(num_users=50, num_items=50, genre_dim=19)
    q = torch.randn(4, 32)
    c = torch.randn(4, 32)
    q = torch.nn.functional.normalize(q, dim=-1)
    c = torch.nn.functional.normalize(c, dim=-1)

    loss = model.compute_loss(q, c)
    assert loss.ndim == 0
    assert float(loss.item()) > 0.0


def test_two_tower_trainer_epoch_convergence() -> None:
    """Verify TwoTowerTrainer decreases loss over training iterations."""
    config = RetrievalConfig(embedding_dim=16, batch_size=8, epochs=3, learning_rate=0.01)
    trainer = TwoTowerTrainer(num_users=20, num_items=20, genre_dim=19, config=config)

    interactions = pd.DataFrame({
        "userId": [1, 2, 3, 4, 5, 6, 7, 8],
        "movieId": [1, 2, 3, 4, 5, 6, 7, 8],
        "label": [1.0] * 8,
    })
    user_feats = pd.DataFrame({
        "userId": list(range(1, 9)),
        "user_interaction_count": [5] * 8,
        "user_avg_rating": [4.0] * 8,
        "user_genre_affinity": [np.zeros(19, dtype=np.float32) for _ in range(8)],
    })
    item_feats = pd.DataFrame({
        "movieId": list(range(1, 9)),
        "item_popularity_score": [0.5] * 8,
        "item_avg_rating": [3.5] * 8,
        "item_genre_vector": [np.zeros(19, dtype=np.float32) for _ in range(8)],
    })

    losses = trainer.fit(interactions, user_feats, item_feats)
    assert len(losses) == 3
    assert all(np.isfinite(loss) for loss in losses)

    # Test item embedding export
    embs, mids = trainer.compute_all_item_embeddings(item_feats)
    assert embs.shape == (8, 16)
    assert mids == list(range(1, 9))

    # Test user query vector
    q_vec = trainer.compute_user_query_vector(1, user_feats)
    assert q_vec.shape == (16,)
    assert np.isclose(np.linalg.norm(q_vec), 1.0, atol=1e-5)


def test_two_tower_out_of_bounds_ids() -> None:
    """Verify towers handle out-of-range user and item IDs gracefully."""
    u_tower = UserQueryTower(num_users=10, genre_dim=19, embedding_dim=16)
    i_tower = CandidateItemTower(num_items=10, genre_dim=19, embedding_dim=16)

    u_ids = torch.tensor([999, -1, 0], dtype=torch.long)
    u_emb = u_tower(u_ids, torch.ones(3), torch.ones(3) * 3.0, torch.zeros((3, 19)))
    assert u_emb.shape == (3, 16)

    i_ids = torch.tensor([888, -2, 0], dtype=torch.long)
    i_emb = i_tower(i_ids, torch.zeros(3), torch.ones(3) * 3.0, torch.zeros((3, 19)))
    assert i_emb.shape == (3, 16)

