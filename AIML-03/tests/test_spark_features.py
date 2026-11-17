"""Unit tests for Milestone 2: Spark feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.features.spark_pipeline import ALL_GENRES, SparkFeaturePipeline


def test_genre_multi_hot_encoding() -> None:
    """Verify pipe-separated genres are correctly encoded as binary multi-hot vectors."""
    spark_pipeline = SparkFeaturePipeline()
    vec = spark_pipeline.encode_genres_multi_hot("Action|Sci-Fi")
    
    assert isinstance(vec, np.ndarray)
    assert len(vec) == len(ALL_GENRES)
    action_idx = spark_pipeline.genre_to_idx["Action"]
    scifi_idx = spark_pipeline.genre_to_idx["Sci-Fi"]
    comedy_idx = spark_pipeline.genre_to_idx["Comedy"]
    
    assert vec[action_idx] == 1.0
    assert vec[scifi_idx] == 1.0
    assert vec[comedy_idx] == 0.0


def test_interaction_table_click_labels() -> None:
    """Verify click labels are assigned based on rating threshold."""
    spark_pipeline = SparkFeaturePipeline()
    ratings = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [10, 20, 10],
        "rating": [4.0, 2.5, 3.5],
        "timestamp": [100, 200, 300],
    })

    interaction_table = spark_pipeline.build_interaction_table(ratings, click_threshold=3.5)
    assert "label" in interaction_table.columns
    assert "normalized_rating" in interaction_table.columns
    # 4.0 and 3.5 are clicks (1.0), 2.5 is not (0.0)
    assert list(interaction_table["label"]) == [1.0, 0.0, 1.0]


def test_user_history_features() -> None:
    """Verify user historical aggregation computes counts, averages, and genre affinity."""
    spark_pipeline = SparkFeaturePipeline()
    ratings = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [10, 20, 10],
        "rating": [4.0, 5.0, 2.0],
        "timestamp": [100, 200, 300],
    })
    movies = pd.DataFrame({
        "movieId": [10, 20],
        "title": ["M10", "M20"],
        "genres": ["Action", "Drama"],
    })

    user_feats = spark_pipeline.compute_user_history_features(ratings, movies)
    u1 = user_feats[user_feats["userId"] == 1].iloc[0]
    
    assert u1["user_interaction_count"] == 2
    assert u1["user_avg_rating"] == 4.5
    assert u1["user_last_timestamp"] == 200
    assert isinstance(u1["user_genre_affinity"], np.ndarray)
    assert len(u1["user_genre_affinity"]) == len(ALL_GENRES)


def test_item_features_and_popularity() -> None:
    """Verify item popularity scores and genre representation."""
    spark_pipeline = SparkFeaturePipeline()
    ratings = pd.DataFrame({
        "userId": [1, 2, 3, 1],
        "movieId": [10, 10, 10, 20],
        "rating": [4.0, 5.0, 3.0, 2.0],
        "timestamp": [100, 200, 300, 400],
    })
    movies = pd.DataFrame({
        "movieId": [10, 20, 30],  # 30 has 0 ratings in catalog
        "title": ["M10", "M20", "M30"],
        "genres": ["Action", "Comedy", "Horror"],
    })

    item_feats = spark_pipeline.compute_item_features(ratings, movies)
    assert len(item_feats) == 3
    
    m10 = item_feats[item_feats["movieId"] == 10].iloc[0]
    m30 = item_feats[item_feats["movieId"] == 30].iloc[0]

    assert m10["item_rating_count"] == 3
    assert m10["item_popularity_score"] == 1.0  # Most popular
    assert m30["item_rating_count"] == 0
    assert m30["item_popularity_score"] == 0.0


def test_empty_ratings_and_movies_features() -> None:
    """Verify feature pipeline handles empty inputs without crashing."""
    spark_pipeline = SparkFeaturePipeline()
    empty_ratings = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    empty_movies = pd.DataFrame(columns=["movieId", "title", "genres"])

    features = spark_pipeline.run_pipeline(empty_ratings, empty_movies)
    assert len(features.interaction_table) == 0
    assert len(features.user_features) == 0
    assert len(features.item_features) == 0

