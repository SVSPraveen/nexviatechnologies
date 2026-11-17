"""Unit tests for Milestone 6: Cold-start fallbacks and Redis caching."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.config import ServingConfig
from recsys.features.spark_pipeline import ALL_GENRES
from recsys.serving.cache import RecommendationCache
from recsys.serving.cold_start import ColdStartHandler


def test_cold_user_detection() -> None:
    """Verify is_cold_user correctly distinguishes known from unknown users."""
    items_df = pd.DataFrame({
        "movieId": [1, 2],
        "item_rating_count": [10, 5],
        "item_avg_rating": [4.0, 3.5],
        "item_popularity_score": [1.0, 0.5],
        "genres": ["Action", "Comedy"],
    })
    handler = ColdStartHandler(items_df)
    known = {1, 2, 3}

    assert not handler.is_cold_user(1, known)
    assert handler.is_cold_user(999, known)


def test_popularity_fallback_ordering() -> None:
    """Verify popularity fallback returns highest Bayesian scored items."""
    items_df = pd.DataFrame({
        "movieId": [10, 20, 30],
        "item_rating_count": [100, 5, 50],
        "item_avg_rating": [4.5, 2.0, 4.0],
        "item_popularity_score": [1.0, 0.05, 0.5],
        "genres": ["Action", "Comedy", "Drama"],
    })
    handler = ColdStartHandler(items_df)
    recs = handler.get_popularity_fallback(top_n=2)

    assert len(recs) == 2
    assert recs[0][0] == 10  # Movie 10 has highest count and rating
    assert recs[0][1] >= recs[1][1]


def test_content_based_fallback_genre_matching() -> None:
    """Verify content-based fallback selects movies matching cold onboarding genres."""
    genre_vocab = ALL_GENRES
    g_to_idx = {g: i for i, g in enumerate(genre_vocab)}

    vec_action = np.zeros(len(genre_vocab), dtype=np.float32)
    vec_action[g_to_idx["Action"]] = 1.0

    vec_romance = np.zeros(len(genre_vocab), dtype=np.float32)
    vec_romance[g_to_idx["Romance"]] = 1.0

    items_df = pd.DataFrame({
        "movieId": [101, 102],
        "item_rating_count": [10, 10],
        "item_avg_rating": [4.0, 4.0],
        "item_popularity_score": [0.5, 0.5],
        "genres": ["Action", "Romance"],
        "item_genre_vector": [vec_action, vec_romance],
    })
    handler = ColdStartHandler(items_df, genre_vocab=genre_vocab)

    recs = handler.get_content_based_fallback(preferred_genres=["Action"], top_n=2)
    assert recs[0][0] == 101  # Action movie ranked first


def test_redis_caching_lifecycle() -> None:
    """Verify write, read, hit, miss, and cache invalidation in Redis layer."""
    config = ServingConfig(redis_port=6379, cache_ttl_seconds=60)
    cache = RecommendationCache(config)

    user_id = 42
    # Initial read: miss
    cached = cache.get_cached_recommendations(user_id)
    assert cached is None
    assert cache.stats["misses"] >= 1

    # Write recommendations
    sample_recs = [(101, 0.95), (202, 0.88), (303, 0.72)]
    cache.cache_recommendations(user_id, sample_recs)

    # Second read: hit
    cached2 = cache.get_cached_recommendations(user_id)
    assert cached2 is not None
    assert len(cached2) == 3
    assert cached2[0] == (101, 0.95)
    assert cache.stats["hits"] >= 1

    # Invalidate cache
    cache.invalidate_user_cache(user_id)
    cached3 = cache.get_cached_recommendations(user_id)
    assert cached3 is None


def test_redis_connection_failure_failover() -> None:
    """Verify RecommendationCache gracefully handles server disconnection without crashing."""
    config = ServingConfig(redis_port=99999)  # Invalid port
    cache = RecommendationCache(config)
    assert cache.is_fake is True

    # Test that operations work seamlessly on failover instance
    cache.cache_recommendations(123, [(10, 0.9)])
    recs = cache.get_cached_recommendations(123)
    assert recs is not None
    assert recs[0] == (10, 0.9)

