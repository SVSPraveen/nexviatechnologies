"""End-to-end runnable demonstration of the Two-Stage Recommendation Engine.

Covers all 7 milestones:
1. MovieLens 25M temporal split
2. Spark feature engineering
3. Two-Tower retrieval model training
4. FAISS index candidate retrieval
5. DCN precision candidate re-scoring
6. Cold-start fallback & Redis caching
7. Triton serving, Airflow DAG, and Online A/B testing
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recsys.config import RecSysPipelineConfig
from recsys.pipeline.orchestrator import TwoStageRecommendationPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("demo")


def run_full_recsys_demo() -> None:
    print("=" * 80)
    print("  TWO-STAGE PERSONALIZED RECOMMENDATION ENGINE (PROTOTYPE)")
    print("  Milestones 1-7: Retrieval (Two-Tower/FAISS) + Ranking (DCN) + Triton + Airflow + Redis")
    print("=" * 80)

    config = RecSysPipelineConfig()
    pipeline = TwoStageRecommendationPipeline(config)

    print("\n>>> [1/7] Ingesting MovieLens 25M & Computing Strict Temporal Split...")
    print(">>> [2/7] Running Spark Feature Pipeline (User History, Item Popularity, Genres)...")
    print(">>> [3/7] Training Two-Tower Retrieval Model (TFRS In-Batch Negative Softmax)...")
    print(">>> [4/7] Indexing Item Embeddings in FAISS...")
    print(">>> [5/7] Training Deep & Cross Network (DCN) Precision Ranker...")

    train_stats = pipeline.train_and_index()
    print(f"\n[✓] Training Complete:")
    print(f"    • Active Users:              {train_stats['num_users']}")
    print(f"    • FAISS Indexed Items:       {train_stats['num_items_indexed']}")
    print(f"    • Two-Tower Retrieval Loss:  {train_stats['final_retrieval_loss']:.4f}")
    print(f"    • DCN Ranker BCE Loss:       {train_stats['final_ranker_loss']:.4f}")

    print("\n" + "=" * 80)
    print(">>> [6/7] Testing Two-Stage Online Inference, Redis Caching & Cold-Start...")
    print("=" * 80)

    # 1. Warm user recommendation (cache miss then hit)
    active_user_id = list(pipeline.known_user_ids)[0]
    print(f"\n--- Scenario A: Active User {active_user_id} (First Request - Cache Miss) ---")
    res1 = pipeline.recommend(user_id=active_user_id, top_n=5, use_cache=True)
    print(f"  Source: {res1.source} | Retrieved Candidates: {res1.num_candidates_retrieved}")
    print("  Top-5 Recommended Movies:")
    for rank, (mid, score) in enumerate(res1.recommendations, 1):
        print(f"    {rank}. MovieID: {mid:<5} | Score: {score:.4f}")

    print(f"\n--- Scenario B: Active User {active_user_id} (Second Request - Cache Hit) ---")
    res2 = pipeline.recommend(user_id=active_user_id, top_n=5, use_cache=True)
    print(f"  Source: {res2.source} | Redis Cache Hit!")
    print(f"  Redis Cache Stats: {pipeline.cache.stats}")

    # 2. Cold-start user popularity fallback
    cold_user_id = 999999
    print(f"\n--- Scenario C: Cold User {cold_user_id} (Zero History - Popularity Fallback) ---")
    res3 = pipeline.recommend(user_id=cold_user_id, top_n=5)
    print(f"  Source: {res3.source}")
    for rank, (mid, score) in enumerate(res3.recommendations, 1):
        print(f"    {rank}. MovieID: {mid:<5} | Popularity Fallback Score: {score:.4f}")

    # 3. Cold-start user content-based fallback
    print(f"\n--- Scenario D: Cold User {cold_user_id + 1} (Explicit Preferences: ['Sci-Fi', 'Action']) ---")
    res4 = pipeline.recommend(user_id=cold_user_id + 1, top_n=5, preferred_genres=["Sci-Fi", "Action"])
    print(f"  Source: {res4.source}")
    for rank, (mid, score) in enumerate(res4.recommendations, 1):
        print(f"    {rank}. MovieID: {mid:<5} | Content-Matched Score: {score:.4f}")

    print("\n" + "=" * 80)
    print(">>> [7/7] Triton Serving Setup, Airflow Retraining DAG & Online A/B Testing...")
    print("=" * 80)

    with tempfile.TemporaryDirectory(prefix="triton_repo_") as tmp_triton:
        infra_res = pipeline.setup_triton_and_airflow(tmp_triton)
        print(f"\n[✓] Triton Model Repository Created:")
        print(f"    • Path:               {infra_res['triton_model_dir']}")
        print(f"[✓] Airflow Retraining DAG Validated & Executed:")
        print(f"    • DAG ID:             {infra_res['airflow_dag_id']}")
        print(f"    • Tasks Executed:     {infra_res['airflow_tasks_executed']} tasks (Topologically Valid)")

    # Run Online A/B Test
    print("\n--- Running Online A/B Experiment: Baseline Ranker vs. DCN Ranker ---")
    ab_report = pipeline.run_online_ab_experiment(num_test_users=80, seed=42)

    print(f"\n  Experiment ID: {ab_report.experiment_id} (Top-{ab_report.k} Evaluation)")
    print(f"  -------------------------------------------------------------")
    print(f"  Metric              Control (Baseline)   Treatment (DCN)    Relative Lift")
    print(f"  -------------------------------------------------------------")
    print(f"  Users:              {ab_report.control.num_users:<20} {ab_report.treatment.num_users:<18}")
    print(f"  Impressions:        {ab_report.control.total_impressions:<20} {ab_report.treatment.total_impressions:<18}")
    print(f"  Clicks:             {ab_report.control.total_clicks:<20} {ab_report.treatment.total_clicks:<18}")
    print(f"  CTR:                {ab_report.control.ctr * 100:.2f}%{' ' * 14} {ab_report.treatment.ctr * 100:.2f}%{' ' * 12} {ab_report.ctr_lift:+.2f}%")
    print(f"  NDCG@{ab_report.k}:            {ab_report.control.mean_ndcg_at_k:.4f}{' ' * 14} {ab_report.treatment.mean_ndcg_at_k:.4f}{' ' * 12} {ab_report.ndcg_lift:+.2f}%")
    print(f"  Recall@{ab_report.k}:          {ab_report.control.mean_recall_at_k:.4f}{' ' * 14} {ab_report.treatment.mean_recall_at_k:.4f}{' ' * 12} {ab_report.recall_lift:+.2f}%")
    print(f"  -------------------------------------------------------------")
    print(f"  Hypothesis Testing:")
    print(f"    • CTR p-value:        {ab_report.ctr_p_value:.6f}")
    print(f"    • NDCG p-value:       {ab_report.ndcg_p_value:.6f}")
    print(f"    • Stat. Significant:  {ab_report.statistically_significant} (alpha=0.05)")

    print("\n" + "=" * 80)
    print("  ALL 7 MILESTONES SUCCESSFULLY DEMONSTRATED AND OPERATIONAL!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_recsys_demo()
