"""End-to-end two-stage recommendation pipeline orchestrator.

Integrates all 7 milestones:
1. MovieLens 25M temporal split
2. Spark feature extraction
3. Two-Tower neural retrieval
4. FAISS nearest-neighbor indexing
5. DCN precision candidate re-scoring
6. Cold-start fallback and Redis caching
7. Triton serving, Airflow DAG scheduling, and online A/B testing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from recsys.config import RecSysPipelineConfig
from recsys.data.movielens import MovieLensDataPipeline, TemporalDatasetSplit
from recsys.experimentation.ab_testing import ABExperimentReport, OnlineABTestRunner
from recsys.features.spark_pipeline import EngineeredFeatures, SparkFeaturePipeline
from recsys.pipeline.airflow_dag import AirflowRetrainingDAG
from recsys.ranking.dcn_model import BaselineRanker, DCNRankerTrainer
from recsys.retrieval.faiss_index import FaissItemIndex
from recsys.retrieval.two_tower import TwoTowerTrainer
from recsys.serving.cache import RecommendationCache
from recsys.serving.cold_start import ColdStartHandler
from recsys.serving.triton_server import TritonModelRepository, TritonServingClient

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Detailed recommendation outcome for a user request."""
    user_id: int
    recommendations: list[tuple[int, float]]
    source: str  # "redis_cache", "two_stage_dcn", "cold_start_popularity", "cold_start_content"
    num_candidates_retrieved: int


class TwoStageRecommendationPipeline:
    """Main orchestrator coordinating retrieval, ranking, caching, and serving."""

    def __init__(self, config: Optional[RecSysPipelineConfig] = None) -> None:
        self.config = config or RecSysPipelineConfig()
        
        # Pipelines and components
        self.data_pipeline = MovieLensDataPipeline(self.config.data)
        self.spark_pipeline = SparkFeaturePipeline()
        self.cache = RecommendationCache(self.config.serving)
        self.ab_runner = OnlineABTestRunner(self.config.ab_test)
        
        # State initialized upon training / loading
        self.dataset_split: Optional[TemporalDatasetSplit] = None
        self.features: Optional[EngineeredFeatures] = None
        self.retrieval_trainer: Optional[TwoTowerTrainer] = None
        self.faiss_index: Optional[FaissItemIndex] = None
        self.dcn_trainer: Optional[DCNRankerTrainer] = None
        self.baseline_ranker = BaselineRanker()
        self.cold_start_handler: Optional[ColdStartHandler] = None
        self.triton_client: Optional[TritonServingClient] = None
        self.known_user_ids: set[int] = set()

    def train_and_index(
        self,
        ratings_path: Optional[Path | str] = None,
        movies_path: Optional[Path | str] = None,
    ) -> dict[str, Any]:
        """Execute complete offline training and indexing workflow across Milestones 1 to 5."""
        logger.info("=== STEP 1: MovieLens Data Cleaning & Strict Temporal Split ===")
        self.dataset_split = self.data_pipeline.load_and_prepare(ratings_path, movies_path)

        logger.info("=== STEP 2: Spark Feature Engineering & Interaction Tables ===")
        self.features = self.spark_pipeline.run_pipeline(
            ratings=self.dataset_split.train_ratings,
            movies=self.dataset_split.movies,
        )
        self.known_user_ids = set(self.features.user_features["userId"].unique())

        logger.info("=== STEP 3: Two-Tower Retrieval Training (TFRS Specification) ===")
        num_users = int(self.dataset_split.train_ratings["userId"].max())
        num_items = int(self.dataset_split.movies["movieId"].max())
        genre_dim = len(self.features.genre_vocab)

        self.retrieval_trainer = TwoTowerTrainer(
            num_users=num_users,
            num_items=num_items,
            genre_dim=genre_dim,
            config=self.config.retrieval,
        )
        retrieval_losses = self.retrieval_trainer.fit(
            train_interactions=self.features.interaction_table,
            user_features_df=self.features.user_features,
            item_features_df=self.features.item_features,
        )

        logger.info("=== STEP 4: FAISS Vector Indexing of Candidate Items ===")
        item_embs, ordered_mids = self.retrieval_trainer.compute_all_item_embeddings(self.features.item_features)
        self.faiss_index = FaissItemIndex(
            embedding_dim=self.config.retrieval.embedding_dim,
            config=self.config.faiss,
        )
        self.faiss_index.build_index(item_embs, ordered_mids)

        logger.info("=== STEP 5: Deep & Cross Network (DCN) Precision Ranker Training ===")
        self.dcn_trainer = DCNRankerTrainer(
            num_users=num_users,
            num_items=num_items,
            genre_dim=genre_dim,
            config=self.config.ranker,
        )
        ranker_losses = self.dcn_trainer.fit(
            train_df=self.features.interaction_table,
            user_features_df=self.features.user_features,
            item_features_df=self.features.item_features,
        )

        # Initialize cold start handler
        self.cold_start_handler = ColdStartHandler(
            item_features_df=self.features.item_features,
            genre_vocab=self.features.genre_vocab,
        )

        # Initialize Triton client with DCN local inference hook
        self.triton_client = TritonServingClient(
            config=self.config.serving,
            local_ranker_fn=lambda uid, cids: self.dcn_trainer.rank_candidates(
                uid, cids, self.features.user_features, self.features.item_features
            ),
        )

        # Invalidate old cache on newly trained model deployment
        self.cache.flush_all()

        return {
            "num_users": len(self.known_user_ids),
            "num_items_indexed": self.faiss_index.total_items,
            "final_retrieval_loss": retrieval_losses[-1] if retrieval_losses else 0.0,
            "final_ranker_loss": ranker_losses[-1] if ranker_losses else 0.0,
        }

    def recommend(
        self,
        user_id: int,
        top_n: Optional[int] = None,
        preferred_genres: Optional[list[str]] = None,
        use_cache: bool = True,
    ) -> RecommendationResult:
        """Online recommendation inference serving path with Redis caching and fallback."""
        k = top_n or self.config.serving.top_n_recommendations

        # 1. Check Redis Cache
        if use_cache:
            cached = self.cache.get_cached_recommendations(user_id)
            if cached is not None:
                return RecommendationResult(
                    user_id=user_id,
                    recommendations=cached[:k],
                    source="redis_cache",
                    num_candidates_retrieved=len(cached),
                )

        # 2. Check for Cold-Start User
        assert self.cold_start_handler is not None, "Pipeline must be trained or loaded before serving."
        if self.cold_start_handler.is_cold_user(user_id, self.known_user_ids):
            if preferred_genres:
                recs = self.cold_start_handler.get_content_based_fallback(preferred_genres, top_n=k)
                source = "cold_start_content"
            else:
                recs = self.cold_start_handler.get_popularity_fallback(top_n=k)
                source = "cold_start_popularity"

            if use_cache:
                self.cache.cache_recommendations(user_id, recs)

            return RecommendationResult(
                user_id=user_id,
                recommendations=recs,
                source=source,
                num_candidates_retrieved=len(recs),
            )

        # 3. Stage 1: Retrieval via Two-Tower + FAISS
        assert self.retrieval_trainer is not None and self.faiss_index is not None and self.features is not None
        query_vec = self.retrieval_trainer.compute_user_query_vector(user_id, self.features.user_features)
        candidates = self.faiss_index.search_candidates(query_vec, k=self.config.faiss.top_k_candidates)
        candidate_ids = [mid for mid, _ in candidates]

        # 4. Stage 2: Precision Ranking via Triton / DCN
        assert self.triton_client is not None
        ranked_candidates = self.triton_client.infer_rankings(user_id, candidate_ids)
        final_recs = ranked_candidates[:k]

        # 5. Cache results into Redis
        if use_cache:
            self.cache.cache_recommendations(user_id, final_recs)

        return RecommendationResult(
            user_id=user_id,
            recommendations=final_recs,
            source="two_stage_dcn",
            num_candidates_retrieved=len(candidate_ids),
        )

    def run_online_ab_experiment(
        self,
        num_test_users: int = 100,
        seed: int = 42,
    ) -> ABExperimentReport:
        """Run simulated online A/B test comparing Baseline Ranker vs. DCN Ranker."""
        assert self.dataset_split is not None and self.features is not None and self.faiss_index is not None
        
        # Test split ground truth interactions
        test_df = self.dataset_split.test_ratings
        ground_truth: dict[int, dict[int, float]] = {}
        for uid, grp in test_df.groupby("userId"):
            ground_truth[int(uid)] = dict(zip(grp["movieId"], grp["rating"]))

        test_uids = list(ground_truth.keys())[:num_test_users]
        if not test_uids:
            test_uids = list(self.known_user_ids)[:num_test_users]

        # Precompute recommendations for Control (Baseline) and Treatment (DCN)
        control_recs: dict[int, list[int]] = {}
        treatment_recs: dict[int, list[int]] = {}
        item_map = self.features.item_features.set_index("movieId").to_dict(orient="index")
        user_map = self.features.user_features.set_index("userId").to_dict(orient="index")

        for uid in test_uids:
            # Stage 1 Retrieval for user
            query_vec = self.retrieval_trainer.compute_user_query_vector(uid, self.features.user_features)
            candidates = self.faiss_index.search_candidates(query_vec, k=self.config.faiss.top_k_candidates)
            cids = [mid for mid, _ in candidates]

            # Variant A: Baseline heuristic ranker
            cand_features = [item_map.get(cid, {}) for cid in cids]
            u_feat = user_map.get(uid, {})
            base_scores = self.baseline_ranker.score_candidates(u_feat, cand_features)
            paired_base = sorted(zip(cids, base_scores), key=lambda x: x[1], reverse=True)
            control_recs[uid] = [mid for mid, _ in paired_base]

            # Variant B: DCN Deep Ranker
            dcn_ranked = self.dcn_trainer.rank_candidates(uid, cids, self.features.user_features, self.features.item_features)
            treatment_recs[uid] = [mid for mid, _ in dcn_ranked]

        report = self.ab_runner.run_experiment(
            test_user_ids=test_uids,
            user_recommendations_control=control_recs,
            user_recommendations_treatment=treatment_recs,
            ground_truth_interactions=ground_truth,
            seed=seed,
        )
        return report

    def setup_triton_and_airflow(self, repo_path: Path | str) -> dict[str, Any]:
        """Configure Triton Model Repository and validate Airflow Retraining DAG."""
        triton_repo = TritonModelRepository(repo_path)
        triton_dir = triton_repo.export_repository_structure(
            retrieval_model=self.retrieval_trainer.model if self.retrieval_trainer else None,
            dcn_model=self.dcn_trainer.model if self.dcn_trainer else None,
        )

        dag = AirflowRetrainingDAG()
        dag.validate_dag()
        dag_execution = dag.execute_all_tasks()

        # Invalidate old cache on newly trained model deployment
        self.cache.flush_all()

        return {
            "triton_model_dir": str(triton_dir),
            "airflow_dag_id": dag.dag_id,
            "airflow_tasks_executed": len(dag_execution),
            "airflow_dag_valid": True,
        }
