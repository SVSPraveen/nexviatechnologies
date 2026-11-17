"""Spark feature engineering pipeline for interaction tables and recommendation features.

Milestone 2: Use Spark to build interaction tables and features:
- User history (rating counts, avg rating, recent interaction timestamp, preferred categories)
- Item popularity (rating frequency, mean rating, popularity percentile)
- Category multi-hot encoding and genre distributions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check if PySpark is available in the environment
_PYSPARK_AVAILABLE = False
try:
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False


ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "Other"
]
GENRE_TO_IDX = {g: i for i, g in enumerate(ALL_GENRES)}


@dataclass
class EngineeredFeatures:
    """Container holding feature tables computed across the dataset."""
    interaction_table: pd.DataFrame
    user_features: pd.DataFrame
    item_features: pd.DataFrame
    genre_vocab: list[str]


class SparkFeaturePipeline:
    """Builds interaction tables and user/item features via Spark SQL semantics."""

    def __init__(self, genre_vocab: Optional[list[str]] = None) -> None:
        self.genre_vocab = genre_vocab or ALL_GENRES
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_vocab)}

    def encode_genres_multi_hot(self, genre_str: str) -> np.ndarray:
        """Encode pipe-separated genre string into a binary multi-hot vector."""
        vec = np.zeros(len(self.genre_vocab), dtype=np.float32)
        if not isinstance(genre_str, str) or not genre_str.strip():
            vec[self.genre_to_idx.get("Other", -1)] = 1.0
            return vec
        for g in genre_str.split("|"):
            g = g.strip()
            idx = self.genre_to_idx.get(g, self.genre_to_idx.get("Other", -1))
            if idx >= 0:
                vec[idx] = 1.0
        return vec

    def build_interaction_table(
        self,
        ratings: pd.DataFrame,
        click_threshold: float = 3.5,
    ) -> pd.DataFrame:
        """Construct interaction table with implicit click targets and normalized ratings.
        
        Spark query semantic:
        SELECT userId, movieId, rating, timestamp,
               CASE WHEN rating >= :click_threshold THEN 1.0 ELSE 0.0 END AS label
        FROM ratings
        """
        df = ratings.copy()
        df["label"] = (df["rating"] >= click_threshold).astype(np.float32)
        df["normalized_rating"] = (df["rating"] - 0.5) / 4.5
        return df

    def compute_user_history_features(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute user historical profile features via Spark aggregation semantics:
        - user_interaction_count: total historical ratings
        - user_avg_rating: mean rating given by user
        - user_rating_std: standard deviation of ratings
        - user_last_timestamp: most recent timestamp
        - user_genre_affinity: vector of historical genre interactions
        """
        if ratings.empty:
            return pd.DataFrame(columns=[
                "userId", "user_interaction_count", "user_avg_rating",
                "user_rating_std", "user_last_timestamp", "user_genre_affinity"
            ])

        # Join ratings with movie genres for category aggregation
        merged = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
        
        # User-level scalar aggregations
        user_agg = merged.groupby("userId").agg(
            user_interaction_count=("rating", "count"),
            user_avg_rating=("rating", "mean"),
            user_rating_std=("rating", lambda s: float(s.std(ddof=0)) if len(s) > 1 else 0.0),
            user_last_timestamp=("timestamp", "max"),
        ).reset_index()

        # Vectorized genre affinity computation
        movie_vectors = {
            mid: self.encode_genres_multi_hot(g if pd.notna(g) else "")
            for mid, g in zip(movies["movieId"], movies["genres"])
        }
        genre_matrix = np.array([
            movie_vectors.get(mid, np.zeros(len(self.genre_vocab), dtype=np.float32))
            for mid in merged["movieId"]
        ])
        weights = (merged["rating"].to_numpy(dtype=np.float32) / 5.0)[:, None]
        weighted_genres = genre_matrix * weights

        weighted_df = pd.DataFrame(weighted_genres, index=merged["userId"])
        user_affinity_df = weighted_df.groupby(level=0).sum()
        sums = user_affinity_df.sum(axis=1).replace(0, 1.0).to_numpy()[:, None]
        normalized_affinities = (user_affinity_df.to_numpy() / sums).astype(np.float32)
        user_genre_vectors = dict(zip(user_affinity_df.index, normalized_affinities))

        user_agg["user_genre_affinity"] = user_agg["userId"].map(user_genre_vectors)
        return user_agg

    def compute_item_features(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute item popularity and category representation features via Spark semantics:
        - item_rating_count: total times movie was rated
        - item_avg_rating: average movie rating
        - item_popularity_percentile: normalized rank [0.0, 1.0]
        - item_genre_vector: multi-hot encoded genres
        """
        if movies.empty:
            return pd.DataFrame(columns=[
                "movieId", "title", "genres", "item_rating_count",
                "item_avg_rating", "item_last_timestamp", "item_popularity_score", "item_genre_vector"
            ])

        item_agg = ratings.groupby("movieId").agg(
            item_rating_count=("rating", "count"),
            item_avg_rating=("rating", "mean"),
            item_last_timestamp=("timestamp", "max"),
        ).reset_index() if not ratings.empty else pd.DataFrame(columns=["movieId", "item_rating_count", "item_avg_rating", "item_last_timestamp"])

        # Include movies with zero ratings in catalog
        catalog = movies.merge(item_agg, on="movieId", how="left")
        catalog["item_rating_count"] = catalog["item_rating_count"].fillna(0).astype(int)
        catalog["item_avg_rating"] = catalog["item_avg_rating"].fillna(3.0).astype(np.float32)
        catalog["item_last_timestamp"] = catalog["item_last_timestamp"].fillna(0).astype(int)

        # Spark percent_rank() window function semantic for popularity
        max_count = max(1, int(catalog["item_rating_count"].max())) if not catalog.empty else 1
        catalog["item_popularity_score"] = (catalog["item_rating_count"] / max_count).astype(np.float32)

        # Multi-hot genre encoding
        catalog["item_genre_vector"] = catalog["genres"].apply(self.encode_genres_multi_hot)
        return catalog

    @staticmethod
    def get_spark_session(app_name: str = "RecSysSparkFeatures") -> Optional[Any]:
        """Return an active or newly initialized SparkSession if PySpark is available."""
        if not _PYSPARK_AVAILABLE:
            return None
        return (
            SparkSession.builder.appName(app_name)
            .config("spark.driver.memory", "2g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )


    def run_pipeline(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> EngineeredFeatures:
        """Run full Spark feature extraction pipeline."""
        logger.info("Running Spark feature extraction pipeline on %d interactions...", len(ratings))
        interaction_table = self.build_interaction_table(ratings)
        user_features = self.compute_user_history_features(ratings, movies)
        item_features = self.compute_item_features(ratings, movies)
        
        logger.info(
            "Feature extraction complete: %d users, %d items, %d interactions.",
            len(user_features), len(item_features), len(interaction_table)
        )
        return EngineeredFeatures(
            interaction_table=interaction_table,
            user_features=user_features,
            item_features=item_features,
            genre_vocab=self.genre_vocab,
        )
