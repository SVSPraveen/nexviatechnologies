"""Airflow retraining DAG and pipeline workflow.

Milestone 7: Schedule retraining with Airflow.
Defines tasks for data ingestion, Spark feature engineering, Two-Tower retrieval training,
FAISS indexing, DCN ranking training, evaluation gating, Triton deployment, and Redis cache warming.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Check if Apache Airflow is installed
_AIRFLOW_AVAILABLE = False
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except (ImportError, Exception):
    _AIRFLOW_AVAILABLE = False



@dataclass
class AirflowTask:
    """Represents an Airflow DAG task operator."""
    task_id: str
    python_callable: Optional[Callable[..., Any]] = None
    upstream_task_ids: set[str] = field(default_factory=set)
    downstream_task_ids: set[str] = field(default_factory=set)

    def set_downstream(self, task: AirflowTask) -> None:
        self.downstream_task_ids.add(task.task_id)
        task.upstream_task_ids.add(self.task_id)

    def __rshift__(self, other: AirflowTask | list[AirflowTask]) -> AirflowTask | list[AirflowTask]:
        if isinstance(other, list):
            for t in other:
                self.set_downstream(t)
            return other
        self.set_downstream(other)
        return other


class AirflowRetrainingDAG:
    """Airflow DAG specification for scheduled model retraining."""

    def __init__(
        self,
        dag_id: str = "recsys_retraining_pipeline",
        schedule_interval: str = "@daily",
        start_date: Optional[datetime] = None,
        retries: int = 2,
        retry_delay_minutes: int = 5,
    ) -> None:
        self.dag_id = dag_id
        self.schedule_interval = schedule_interval
        self.start_date = start_date or datetime(2026, 1, 1)
        self.default_args = {
            "owner": "recsys_team",
            "depends_on_past": False,
            "retries": retries,
            "retry_delay": timedelta(minutes=retry_delay_minutes),
        }
        self.tasks: dict[str, AirflowTask] = {}
        self._build_dag_graph()

    def add_task(self, task_id: str, python_callable: Optional[Callable[..., Any]] = None) -> AirflowTask:
        task = AirflowTask(task_id=task_id, python_callable=python_callable)
        self.tasks[task_id] = task
        return task

    def _build_dag_graph(self) -> None:
        """Construct tasks and dependencies for full recommender retraining pipeline."""
        # 1. Ingestion & Temporal Splitting
        t_ingest = self.add_task(
            "data_ingestion_and_cleaning",
            python_callable=lambda: logger.info("DAG Task: Ingesting and temporally splitting MovieLens data..."),
        )

        # 2. Spark Feature Engineering
        t_spark = self.add_task(
            "spark_feature_extraction",
            python_callable=lambda: logger.info("DAG Task: Computing Spark interaction tables & user/item features..."),
        )

        # 3. Two-Tower Retrieval Training
        t_retrieval = self.add_task(
            "train_two_tower_retrieval",
            python_callable=lambda: logger.info("DAG Task: Training Two-Tower retrieval model..."),
        )

        # 4. FAISS Vector Indexing
        t_faiss = self.add_task(
            "build_faiss_vector_index",
            python_callable=lambda: logger.info("DAG Task: Indexing candidate item embeddings in FAISS..."),
        )

        # 5. DCN Precision Ranking Training
        t_dcn = self.add_task(
            "train_dcn_precision_ranker",
            python_callable=lambda: logger.info("DAG Task: Training Deep & Cross Network ranker..."),
        )

        # 6. Evaluation & Promotion Gate
        t_eval = self.add_task(
            "offline_model_evaluation",
            python_callable=lambda: logger.info("DAG Task: Evaluating NDCG@K and Recall@K on test split..."),
        )

        # 7. Triton Model Repository Deployment
        t_deploy = self.add_task(
            "deploy_models_to_triton",
            python_callable=lambda: logger.info("DAG Task: Updating Triton Inference Server model repository..."),
        )

        # 8. Redis Cache Invalidation & Warming
        t_cache = self.add_task(
            "invalidate_and_warm_redis_cache",
            python_callable=lambda: logger.info("DAG Task: Invalidating stale cache and pre-warming top users..."),
        )

        # Build Airflow dependency graph:
        # t_ingest >> t_spark >> [t_retrieval, t_dcn]
        # t_retrieval >> t_faiss >> t_eval
        # t_dcn >> t_eval
        # t_eval >> t_deploy >> t_cache
        t_ingest >> t_spark
        t_spark >> t_retrieval
        t_spark >> t_dcn
        t_retrieval >> t_faiss
        t_faiss >> t_eval
        t_dcn >> t_eval
        t_eval >> t_deploy
        t_deploy >> t_cache

    def validate_dag(self) -> bool:
        """Verify DAG acyclicity (topological sort) and completeness."""
        in_degree = {task_id: len(task.upstream_task_ids) for task_id, task in self.tasks.items()}
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            current = queue.pop(0)
            visited_count += 1
            for down_id in self.tasks[current].downstream_task_ids:
                in_degree[down_id] -= 1
                if in_degree[down_id] == 0:
                    queue.append(down_id)

        if visited_count != len(self.tasks):
            raise ValueError(f"Cycle detected in Airflow DAG {self.dag_id}! Visited {visited_count} of {len(self.tasks)}.")
        return True

    def to_airflow_dag(self) -> Optional[Any]:
        """Convert to official Apache Airflow DAG instance if Apache Airflow is installed."""
        if not _AIRFLOW_AVAILABLE:
            return None
        dag = DAG(
            dag_id=self.dag_id,
            schedule_interval=self.schedule_interval,
            start_date=self.start_date,
            default_args=self.default_args,
            catchup=False,
        )
        airflow_operators: dict[str, Any] = {}
        with dag:
            for tid, t in self.tasks.items():
                op = PythonOperator(
                    task_id=tid,
                    python_callable=t.python_callable or (lambda: None),
                )
                airflow_operators[tid] = op
            for tid, t in self.tasks.items():
                for down_id in t.downstream_task_ids:
                    airflow_operators[tid] >> airflow_operators[down_id]
        return dag

    def execute_all_tasks(self) -> dict[str, str]:
        """Execute tasks in topological order with retries and failure propagation."""
        self.validate_dag()
        execution_log: dict[str, str] = {}
        failed_tasks: set[str] = set()

        # Determine execution order
        in_degree = {task_id: len(task.upstream_task_ids) for task_id, task in self.tasks.items()}
        queue = [tid for tid, deg in in_degree.items() if deg == 0]

        max_retries = int(self.default_args.get("retries", 0))

        while queue:
            task_id = queue.pop(0)
            task = self.tasks[task_id]

            # Check if any upstream task failed
            if any(up in failed_tasks for up in task.upstream_task_ids):
                execution_log[task_id] = "UPSTREAM_FAILED"
                failed_tasks.add(task_id)
            else:
                success = False
                for attempt in range(max_retries + 1):
                    try:
                        if task.python_callable is not None:
                            task.python_callable()
                        success = True
                        break
                    except Exception as exc:
                        logger.warning("Task %s attempt %d/%d failed: %s", task_id, attempt + 1, max_retries + 1, exc)
                if success:
                    execution_log[task_id] = "SUCCESS"
                else:
                    execution_log[task_id] = "FAILED"
                    failed_tasks.add(task_id)

            for down_id in task.downstream_task_ids:
                in_degree[down_id] -= 1
                if in_degree[down_id] == 0:
                    queue.append(down_id)

        return execution_log
