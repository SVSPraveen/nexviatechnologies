"""Integration test for full Two-Stage Recommendation Pipeline."""

from __future__ import annotations

import tempfile

from recsys.config import (
    DataConfig,
    FaissConfig,
    RankerConfig,
    RecSysPipelineConfig,
    RetrievalConfig,
    ServingConfig,
)
from recsys.pipeline.orchestrator import TwoStageRecommendationPipeline


def test_full_two_stage_pipeline_integration() -> None:
    """Verify end-to-end execution of all 7 milestones:
    Data -> Spark -> Two-Tower -> FAISS -> DCN Ranker -> Redis -> Triton & A/B testing.
    """
    config = RecSysPipelineConfig(
        data=DataConfig(min_user_interactions=2, min_item_interactions=2),
        retrieval=RetrievalConfig(embedding_dim=16, epochs=2, batch_size=16),
        faiss=FaissConfig(top_k_candidates=20),
        ranker=RankerConfig(epochs=2, batch_size=16, deep_layer_sizes=[32, 16]),
        serving=ServingConfig(top_n_recommendations=5),
    )
    pipeline = TwoStageRecommendationPipeline(config)

    # 1. Train and Index
    stats = pipeline.train_and_index()
    assert stats["num_users"] > 0
    assert stats["num_items_indexed"] > 0
    assert stats["final_retrieval_loss"] > 0
    assert stats["final_ranker_loss"] > 0

    # 2. Known User Recommendation (Retrieval + DCN Ranking)
    user_id = list(pipeline.known_user_ids)[0]
    rec_res = pipeline.recommend(user_id=user_id, top_n=5, use_cache=True)
    assert rec_res.user_id == user_id
    assert len(rec_res.recommendations) == 5
    assert rec_res.source == "two_stage_dcn"

    # 3. Redis Cache Hit
    cached_res = pipeline.recommend(user_id=user_id, top_n=5, use_cache=True)
    assert cached_res.source == "redis_cache"
    assert cached_res.recommendations == rec_res.recommendations

    # 4. Cold-Start User (Popularity Fallback)
    cold_user_id = 888888
    cold_res = pipeline.recommend(user_id=cold_user_id, top_n=5)
    assert cold_res.source == "cold_start_popularity"
    assert len(cold_res.recommendations) == 5

    # 5. Cold-Start User (Content-based Fallback)
    cold_content_res = pipeline.recommend(user_id=cold_user_id + 1, top_n=5, preferred_genres=["Comedy"])
    assert cold_content_res.source == "cold_start_content"
    assert len(cold_content_res.recommendations) == 5

    # 6. Triton Repository & Airflow Retraining DAG
    with tempfile.TemporaryDirectory() as tmp_dir:
        infra = pipeline.setup_triton_and_airflow(tmp_dir)
        assert infra["airflow_dag_valid"] is True
        assert infra["airflow_tasks_executed"] == 8

    # 7. Online A/B Testing Experiment
    ab_report = pipeline.run_online_ab_experiment(num_test_users=30, seed=42)
    assert ab_report.control.num_users + ab_report.treatment.num_users == 30
    assert ab_report.control.total_impressions > 0
    assert ab_report.treatment.total_impressions > 0
