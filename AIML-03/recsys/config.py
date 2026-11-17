"""Configuration schemas and defaults for the Recommendation Engine Prototype."""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Configuration for data loading and temporal splitting."""
    dataset_name: str = "movielens_25m"
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    # MovieLens 25M temporal split ratios (e.g. 80% train, 10% val, 10% test by timestamp)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42

    @model_validator(mode="after")
    def validate_ratios(self) -> DataConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Split ratios must sum to 1.0; got {total:.4f}")
        return self



class RetrievalConfig(BaseModel):
    """Configuration for Two-Tower Retrieval Model."""
    embedding_dim: int = 32
    user_layers: list[int] = Field(default_factory=lambda: [64, 32])
    item_layers: list[int] = Field(default_factory=lambda: [64, 32])
    learning_rate: float = 0.01
    batch_size: int = 64
    epochs: int = 5
    temperature: float = 0.05


class FaissConfig(BaseModel):
    """Configuration for FAISS candidate indexing."""
    index_type: str = "HNSW"  # Flat or HNSW
    m: int = 16
    ef_construction: int = 64
    ef_search: int = 32
    top_k_candidates: int = 50


class RankerConfig(BaseModel):
    """Configuration for Deep & Cross Network (DCN) Ranker."""
    num_cross_layers: int = 3
    deep_layer_sizes: list[int] = Field(default_factory=lambda: [64, 32])
    embedding_dim: int = 16
    dropout_rate: float = 0.1
    learning_rate: float = 0.005
    batch_size: int = 64
    epochs: int = 5
    l2_reg: float = 1e-4


class ServingConfig(BaseModel):
    """Configuration for serving, Redis caching, and cold-start fallback."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl_seconds: int = 3600
    top_n_recommendations: int = 10
    cold_start_popularity_count: int = 20
    triton_model_name: str = "recsys_two_stage"
    triton_url: str = "localhost:8001"


class ABTestConfig(BaseModel):
    """Configuration for online A/B testing."""
    test_id: str = "exp_ranker_dcn_vs_baseline"
    traffic_split_a: float = 0.50  # 50% Control (Baseline), 50% Treatment (DCN)
    k_eval: int = 10
    significance_alpha: float = 0.05


class RecSysPipelineConfig(BaseModel):
    """Global configuration aggregating all milestone components."""
    data: DataConfig = Field(default_factory=DataConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    faiss: FaissConfig = Field(default_factory=FaissConfig)
    ranker: RankerConfig = Field(default_factory=RankerConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    ab_test: ABTestConfig = Field(default_factory=ABTestConfig)
