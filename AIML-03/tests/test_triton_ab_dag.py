"""Unit tests for Milestone 7: Triton Serving, Airflow Retraining DAG, and Online A/B Testing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from recsys.config import ABTestConfig
from recsys.experimentation.ab_testing import (
    OnlineABTestRunner,
    compute_dcg_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
)
from recsys.pipeline.airflow_dag import AirflowRetrainingDAG
from recsys.serving.triton_server import (
    TritonModelRepository,
    TritonServingClient,
    generate_triton_dcn_config,
    generate_triton_two_tower_config,
)


def test_triton_configs_and_repository() -> None:
    """Verify Triton configuration files and repository layout are generated."""
    conf_retrieval = generate_triton_two_tower_config("test_retrieval")
    assert 'name: "test_retrieval"' in conf_retrieval
    assert "QUERY_EMBEDDING" in conf_retrieval

    conf_dcn = generate_triton_dcn_config("test_dcn")
    assert 'name: "test_dcn"' in conf_dcn
    assert "PROBABILITIES" in conf_dcn

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = TritonModelRepository(tmp_dir)
        repo_path = repo.export_repository_structure()
        assert (repo_path / "retrieval_two_tower" / "config.pbtxt").exists()
        assert (repo_path / "retrieval_two_tower" / "1").is_dir()
        assert (repo_path / "retrieval_two_tower" / "1" / "model.pt").exists()
        assert (repo_path / "ranker_dcn" / "config.pbtxt").exists()
        assert (repo_path / "ranker_dcn" / "1").is_dir()
        assert (repo_path / "ranker_dcn" / "1" / "model.pt").exists()


def test_triton_serving_client_inference_fallback() -> None:
    """Verify Triton client dispatches candidate ranking via hook."""
    mock_ranker = lambda uid, cids: [(cid, 0.99 - idx * 0.1) for idx, cid in enumerate(cids)]
    client = TritonServingClient(local_ranker_fn=mock_ranker)

    res = client.infer_rankings(user_id=1, candidate_ids=[10, 20, 30])
    assert len(res) == 3
    assert res[0][0] == 10
    assert np.isclose(res[0][1], 0.99)


def test_airflow_retraining_dag_validation_and_execution() -> None:
    """Verify Airflow DAG dependencies, topological validity, and execution order."""
    dag = AirflowRetrainingDAG(dag_id="test_dag")
    assert len(dag.tasks) == 8

    # Validate acyclicity
    assert dag.validate_dag() is True

    # Execute all tasks in topological order
    exec_log = dag.execute_all_tasks()
    assert len(exec_log) == 8
    assert all(status == "SUCCESS" for status in exec_log.values())


def test_ndcg_and_recall_metrics() -> None:
    """Verify NDCG@K and Recall@K mathematical formulas."""
    # Perfect ranking test
    recommended = [10, 20, 30]
    ground_truth = {10: 5.0, 20: 3.0, 30: 1.0}
    positives = {10, 20, 30}

    ndcg = compute_ndcg_at_k(recommended, ground_truth, k=3)
    recall = compute_recall_at_k(recommended, positives, k=3)

    assert np.isclose(ndcg, 1.0)
    assert np.isclose(recall, 1.0)

    # Empty edge cases
    assert compute_ndcg_at_k([], {}, k=5) == 0.0
    assert compute_recall_at_k([], set(), k=5) == 0.0


def test_online_ab_experiment_runner() -> None:
    """Verify A/B testing user assignment, CTR, NDCG lift, and statistical tests."""
    config = ABTestConfig(test_id="test_exp_1", traffic_split_a=0.50, k_eval=5)
    runner = OnlineABTestRunner(config)

    # Deterministic assignment
    v1 = runner.assign_variant(101)
    v2 = runner.assign_variant(101)
    assert v1 == v2
    assert v1 in {"Control_A", "Treatment_B"}

    # Simulate experiment
    users = list(range(1, 41))
    ctrl_recs = {uid: [1, 2, 3, 4, 5] for uid in users}
    treat_recs = {uid: [5, 4, 3, 2, 1] for uid in users}
    gt = {uid: {5: 5.0, 4: 4.0 if uid % 2 == 0 else 2.0} for uid in users}

    report = runner.run_experiment(
        test_user_ids=users,
        user_recommendations_control=ctrl_recs,
        user_recommendations_treatment=treat_recs,
        ground_truth_interactions=gt,
        seed=42,
    )

    assert report.k == 5
    assert report.control.num_users + report.treatment.num_users == 40
    assert report.control.total_impressions > 0
    assert report.treatment.total_impressions > 0
    assert 0.0 <= report.ctr_p_value <= 1.0
    assert 0.0 <= report.ndcg_p_value <= 1.0


def test_airflow_retraining_dag_failure_and_retries() -> None:
    """Verify task retry attempts and downstream UPSTREAM_FAILED propagation upon failure."""
    dag = AirflowRetrainingDAG(dag_id="failure_test_dag", retries=2)
    attempts = [0]

    def failing_callable():
        attempts[0] += 1
        raise RuntimeError("Simulated transient failure")

    # Override root task to fail
    dag.tasks["data_ingestion_and_cleaning"].python_callable = failing_callable

    exec_log = dag.execute_all_tasks()
    assert attempts[0] == 3  # Initial + 2 retries
    assert exec_log["data_ingestion_and_cleaning"] == "FAILED"
    # All downstream tasks should be marked UPSTREAM_FAILED
    assert exec_log["spark_feature_extraction"] == "UPSTREAM_FAILED"
    assert exec_log["train_two_tower_retrieval"] == "UPSTREAM_FAILED"


def test_online_ab_zero_variance_ndcg() -> None:
    """Verify A/B testing handles zero variance NDCG without RuntimeWarning or NaN."""
    config = ABTestConfig(test_id="test_exp_zero_var", k_eval=5)
    runner = OnlineABTestRunner(config)

    users = [1, 2, 3, 4]
    ctrl_recs = {uid: [10, 20] for uid in users}
    treat_recs = {uid: [10, 20] for uid in users}
    gt = {uid: {99: 5.0} for uid in users}  # Zero hits

    report = runner.run_experiment(users, ctrl_recs, treat_recs, gt, seed=42)
    assert report.control.mean_ndcg_at_k == 0.0
    assert report.treatment.mean_ndcg_at_k == 0.0
    assert report.ndcg_p_value == 1.0
    assert not np.isnan(report.ndcg_p_value)

