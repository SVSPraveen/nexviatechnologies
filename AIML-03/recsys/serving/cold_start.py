"""Cold-start user and item fallback handlers.

Milestone 6: Handle cold-start users with popularity and content-based fallbacks.
Provides robust fallbacks when users have zero historical interactions or
explicit genre-based cold onboarding.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from recsys.features.spark_pipeline import ALL_GENRES

logger = logging.getLogger(__name__)


class ColdStartHandler:
    """Handles recommendation requests for cold-start users and items."""

    def __init__(
        self,
        item_features_df: pd.DataFrame,
        genre_vocab: Optional[list[str]] = None,
    ) -> None:
        self.item_features_df = item_features_df.copy()
        self.genre_vocab = genre_vocab or ALL_GENRES
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_vocab)}
        
        # Precompute popularity fallback ranking
        # Bayesian average score: (v * R + m * C) / (v + m)
        v = self.item_features_df["item_rating_count"].astype(float)
        r = self.item_features_df["item_avg_rating"].astype(float)
        c = r.mean() if not r.empty else 3.0
        m = 5.0
        self.item_features_df["bayesian_score"] = (v * r + m * c) / (v + m)
        self.popularity_ranked_items = (
            self.item_features_df.sort_values(
                by=["bayesian_score", "item_rating_count"],
                ascending=[False, False],
            )["movieId"].tolist()
        )

    def is_cold_user(self, user_id: int, known_user_ids: set[int]) -> bool:
        """Determine if user is a cold-start user (never seen in training interactions)."""
        return user_id not in known_user_ids

    def get_popularity_fallback(self, top_n: int = 10) -> list[tuple[int, float]]:
        """Return globally popular, highly-rated items for new users with zero history."""
        top_ids = self.popularity_ranked_items[:top_n]
        results: list[tuple[int, float]] = []
        for rank, mid in enumerate(top_ids):
            # Rank score decaying from 1.0 down to 0.5
            score = 1.0 - (rank / max(1, top_n)) * 0.5
            results.append((int(mid), float(score)))
        return results

    def get_content_based_fallback(
        self,
        preferred_genres: list[str],
        top_n: int = 10,
    ) -> list[tuple[int, float]]:
        """Return content-based recommendations matched against cold onboarding genre preferences."""
        if not preferred_genres:
            return self.get_popularity_fallback(top_n=top_n)

        # Build query genre vector
        query_vec = np.zeros(len(self.genre_vocab), dtype=np.float32)
        for g in preferred_genres:
            idx = self.genre_to_idx.get(g, -1)
            if idx >= 0:
                query_vec[idx] = 1.0

        scores: list[tuple[int, float]] = []
        for _, row in self.item_features_df.iterrows():
            mid = int(row["movieId"])
            item_vec = row.get("item_genre_vector", np.zeros(len(self.genre_vocab), dtype=np.float32))
            
            # Genre overlap cosine similarity
            norm_q = np.linalg.norm(query_vec)
            norm_i = np.linalg.norm(item_vec)
            if norm_q > 0 and norm_i > 0:
                genre_sim = float(np.dot(query_vec, item_vec) / (norm_q * norm_i))
            else:
                genre_sim = 0.0

            # Composite content score: 70% genre match + 30% popularity/rating
            pop_score = float(row.get("item_popularity_score", 0.0))
            composite_score = 0.70 * genre_sim + 0.30 * pop_score
            scores.append((mid, composite_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
