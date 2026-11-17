"""Unit tests for Milestone 1: MovieLens data pipeline, cleaning, and temporal split."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.config import DataConfig
from recsys.data.movielens import (
    MovieLensDataPipeline,
    TemporalDatasetSplit,
    generate_sample_movielens_25m,
)


def test_synthetic_movielens_schema() -> None:
    """Verify synthetic MovieLens sample adheres to 25M schema."""
    ratings, movies = generate_sample_movielens_25m(num_users=50, num_movies=20, num_interactions=200, seed=123)
    
    assert set(ratings.columns) == {"userId", "movieId", "rating", "timestamp"}
    assert set(movies.columns) == {"movieId", "title", "genres"}
    assert len(ratings) > 0
    assert len(movies) == 20
    assert ratings["rating"].between(0.5, 5.0).all()
    assert (ratings["timestamp"] > 0).all()


def test_data_cleaner_filtering() -> None:
    """Verify cleaner removes nulls, invalid ranges, and low-frequency users/items."""
    config = DataConfig(min_user_interactions=2, min_item_interactions=2)
    pipeline = MovieLensDataPipeline(config)

    raw_ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2, 3, 4],
        "movieId": [10, 20, 10, 20, 10, 999],  # 999 not in catalog; user 3 has 1 interaction, user 4 has 999
        "rating": [4.0, 5.0, 4.5, 3.5, 2.0, 5.0],
        "timestamp": [1000, 2000, 3000, 4000, 5000, 6000],
    })
    movies = pd.DataFrame({
        "movieId": [10, 20],
        "title": ["M10", "M20"],
        "genres": ["Action", "Comedy"],
    })

    cleaned = pipeline.clean_ratings(raw_ratings, movies)
    # Movie 999 filtered out; users with < 2 interactions iteratively filtered
    assert 999 not in cleaned["movieId"].values
    assert set(cleaned["userId"].unique()) == {1, 2}
    assert len(cleaned) == 4


def test_strict_temporal_split_zero_leakage() -> None:
    """Verify temporal split strictly partitions by timestamp with zero leakage."""
    config = DataConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_user_interactions=1, min_item_interactions=1)
    pipeline = MovieLensDataPipeline(config)

    ratings, movies = generate_sample_movielens_25m(num_users=100, num_movies=30, num_interactions=1000, seed=42)
    split = pipeline.temporal_split(ratings, movies)

    assert isinstance(split, TemporalDatasetSplit)
    assert len(split.train_ratings) > 0
    assert len(split.val_ratings) > 0
    assert len(split.test_ratings) > 0

    # Strict temporal boundary check: max(train) <= min(val) <= max(val) <= min(test)
    max_train_ts = split.train_ratings["timestamp"].max()
    min_val_ts = split.val_ratings["timestamp"].min()
    max_val_ts = split.val_ratings["timestamp"].max()
    min_test_ts = split.test_ratings["timestamp"].min()

    assert max_train_ts <= min_val_ts, f"Leakage: train max {max_train_ts} > val min {min_val_ts}"
    assert max_val_ts <= min_test_ts, f"Leakage: val max {max_val_ts} > test min {min_test_ts}"


def test_empty_ratings_split_raises() -> None:
    """Verify ValueError is raised if attempting to split an empty dataframe."""
    pipeline = MovieLensDataPipeline()
    empty_df = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    movies_df = pd.DataFrame(columns=["movieId", "title", "genres"])
    with pytest.raises(ValueError, match="Cannot split empty"):
        pipeline.temporal_split(empty_df, movies_df)


def test_invalid_split_ratios_raise() -> None:
    """Verify DataConfig raises ValueError if split ratios do not sum to 1.0."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        DataConfig(train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)

