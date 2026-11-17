"""Unit tests for Milestone 5: Deep & Cross Network (DCN) ranking model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from recsys.config import RankerConfig
from recsys.ranking.dcn_model import (
    BaselineRanker,
    CrossNetwork,
    CrossNetworkLayer,
    DCNRankerTrainer,
    DeepCrossNetwork,
)


def test_cross_network_layer_forward() -> None:
    """Verify single CrossNetworkLayer computes x_{l+1} = x0 * (xl^T w) + b + xl."""
    layer = CrossNetworkLayer(input_dim=5)
    x0 = torch.randn(3, 5)
    xl = torch.randn(3, 5)

    out = layer(x0, xl)
    assert out.shape == (3, 5)
    assert not torch.isnan(out).any()


def test_cross_network_stack() -> None:
    """Verify multi-layer CrossNetwork preserves input dimension."""
    cnet = CrossNetwork(input_dim=8, num_layers=3)
    x0 = torch.randn(4, 8)
    out = cnet(x0)
    assert out.shape == (4, 8)


def test_deep_cross_network_forward_predictions() -> None:
    """Verify DCN forward pass produces probabilities in [0.0, 1.0]."""
    config = RankerConfig(embedding_dim=8, num_cross_layers=2, deep_layer_sizes=[16, 8])
    dcn = DeepCrossNetwork(num_users=20, num_items=20, genre_dim=19, config=config)

    u_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    i_ids = torch.tensor([10, 15, 20], dtype=torch.long)
    dense = torch.randn(3, 5)

    probs = dcn(u_ids, i_ids, dense)
    assert probs.shape == (3,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_dcn_ranker_trainer_fit_and_candidate_ranking() -> None:
    """Verify DCNRankerTrainer trains and ranks candidate items descending."""
    config = RankerConfig(epochs=2, batch_size=4, learning_rate=0.01)
    trainer = DCNRankerTrainer(num_users=10, num_items=10, genre_dim=19, config=config)

    train_df = pd.DataFrame({
        "userId": [1, 2, 1, 2, 1, 2],
        "movieId": [1, 2, 3, 4, 5, 6],
        "label": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    })
    user_feats = pd.DataFrame({
        "userId": [1, 2],
        "user_interaction_count": [10, 5],
        "user_avg_rating": [4.5, 3.0],
        "user_genre_affinity": [np.zeros(19, dtype=np.float32) for _ in range(2)],
    })
    item_feats = pd.DataFrame({
        "movieId": list(range(1, 7)),
        "item_popularity_score": [0.8, 0.2, 0.9, 0.1, 0.7, 0.5],
        "item_avg_rating": [4.0, 2.5, 4.5, 3.0, 4.2, 3.8],
        "item_genre_vector": [np.zeros(19, dtype=np.float32) for _ in range(6)],
    })

    losses = trainer.fit(train_df, user_feats, item_feats)
    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)

    # Re-rank candidates
    candidate_ids = [2, 1, 3]
    ranked = trainer.rank_candidates(user_id=1, candidate_ids=candidate_ids, user_features_df=user_feats, item_features_df=item_feats)

    assert len(ranked) == 3
    # Check descending order
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_baseline_ranker_scoring() -> None:
    """Verify baseline heuristic ranker scores candidates in [0, 1]."""
    ranker = BaselineRanker()
    u_feat = {"user_genre_affinity": np.zeros(19)}
    candidates = [
        {"movieId": 1, "item_popularity_score": 0.9, "item_avg_rating": 4.5, "item_genre_vector": np.zeros(19)},
        {"movieId": 2, "item_popularity_score": 0.2, "item_avg_rating": 2.5, "item_genre_vector": np.zeros(19)},
    ]

    scores = ranker.score_candidates(u_feat, candidates)
    assert len(scores) == 2
    assert scores[0] > scores[1]  # Popular item scores higher


def test_dcn_out_of_bounds_ids() -> None:
    """Verify DCN handles out-of-range user and movie IDs gracefully without IndexError."""
    config = RankerConfig(embedding_dim=8, num_cross_layers=2, deep_layer_sizes=[16, 8])
    dcn = DeepCrossNetwork(num_users=10, num_items=10, genre_dim=19, config=config)

    # Pass out-of-range user ID 999 and item ID 888
    u_ids = torch.tensor([999, -5, 0], dtype=torch.long)
    i_ids = torch.tensor([888, -1, 0], dtype=torch.long)
    dense = torch.zeros((3, 5), dtype=torch.float32)

    probs = dcn(u_ids, i_ids, dense)
    assert probs.shape == (3,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

