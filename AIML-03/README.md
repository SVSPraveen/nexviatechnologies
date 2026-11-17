# AIML-03: Two-Stage Personalized Recommendation Engine Prototype

A production-grade prototype of a two-stage personalized recommender system inspired by modern industrial architectures (Netflix, Amazon). Stage one rapidly narrows down large catalogs to a few hundred candidates using dense neural retrieval and approximate nearest-neighbor search; stage two scores and re-ranks them with a Deep & Cross Network (DCN).

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       MovieLens 25M Interactions      |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Milestone 1: Strict Temporal Split  |
                      |   (Train: 80% | Val: 10% | Test: 10%) |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Milestone 2: Spark Feature Pipeline   |
                      | - User history & Genre Affinities     |
                      | - Item Popularity & Multi-hot Genres  |
                      +---------------------------------------+
                                          |
                  +-----------------------+-----------------------+
                  |                                               |
                  v                                               v
+------------------------------------+          +------------------------------------+
| Milestone 3: Two-Tower Retrieval   |          | Milestone 5: Deep & Cross Network  |
| - Query Tower (User + Affinities)  |          | - Cross Net (Explicit interactions)|
| - Candidate Tower (Item + Genres)  |          | - Deep MLP (Non-linear features)   |
| - In-Batch Negative Softmax Loss   |          | - Binary Cross-Entropy Head        |
+------------------------------------+          +------------------------------------+
                  |                                               |
                  v                                               |
+------------------------------------+                            |
| Milestone 4: FAISS Vector Index    |                            |
| - IndexFlatIP & IndexHNSWFlat      |                            |
| - Top-50 Fast Candidate Retrieval  |                            |
+------------------------------------+                            |
                  |                                               |
                  +-----------------------+-----------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Milestone 6: Cold-Start & Redis Cache |
                      | - Popularity & Content Fallbacks      |
                      | - Redis TTL Write-Through Caching     |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Milestone 7: Serving & Experimentation|
                      | - Triton Inference Server Repository  |
                      | - Airflow Retraining DAG (8 tasks)    |
                      | - Online A/B Testing (NDCG, Recall,   |
                      |   CTR & Hypothesis Significance Tests)|
                      +---------------------------------------+
```

---

## Detailed Milestone Implementation

### Milestone 1: MovieLens Ingestion & Strict Temporal Split
- **Dataset**: MovieLens 25M schema (`userId`, `movieId`, `rating`, `timestamp`, `title`, `genres`).
- **Cleaning**: Drops invalid/null entries, bounds ratings in $[0.5, 5.0]$, and applies $k$-core minimum interaction filtering.
- **Leakage Prevention**: Strictly splits interaction tables chronologically into train, validation, and test sets. $\max(t_{\text{train}}) \le \min(t_{\text{val}})$ and $\max(t_{\text{val}}) \le \min(t_{\text{test}})$, mathematically preventing future information from leaking into past features.

### Milestone 2: Apache Spark Feature Engineering
- **Interaction Tables**: Generates implicit interaction labels ($\text{rating} \ge 3.5 \implies \text{click}$) and normalized continuous ratings.
- **User Profile Features**: Computes rolling user interaction counts, average ratings, variance, recency timestamps, and historical genre affinity distribution vectors.
- **Item Profile Features**: Computes item rating volume, average ratings, popularity percentile scores, and multi-hot genre representations.

### Milestone 3: Two-Tower Retrieval Model
- **User Query Tower**: Deep projection network transforming sparse user identity, interaction volume, average rating, and genre affinity into an $L_2$-normalized embedding vector $u \in \mathbb{R}^d$.
- **Candidate Item Tower**: Deep projection network transforming movie identity, popularity score, and multi-hot genre vector into an $L_2$-normalized embedding vector $v \in \mathbb{R}^d$.
- **Objective**: In-batch negative sampled softmax cross-entropy loss matching TensorFlow Recommenders (`tfrs.tasks.Retrieval`).

### Milestone 4: FAISS Vector Indexing
- **Indices**: Indexes all catalog candidate vectors into FAISS `IndexFlatIP` (exact cosine inner product) and `IndexHNSWFlat` (hierarchical navigable small world graph).
- **Candidate Generation**: Retrieves top-$K$ candidate items for a user query vector in sub-millisecond latency.
- **Persistence**: Supports cryptographic atomic serialization and reloading.

### Milestone 5: Deep & Cross Network (DCN) Precision Ranker
- **Cross Network**: Explicit feature crossing across $L$ layers:
  $$x_{l+1} = x_0 (x_l^T w_l) + b_l + x_l$$
  capturing bounded-degree polynomial interactions without combinatorial explosion.
- **Deep Network**: Deep feedforward MLP with LayerNorm, ReLU activations, and Dropout.
- **Re-scoring**: Scores all candidates retrieved from Stage 1 and re-ranks them by calibrated click/relevance probability.

### Milestone 6: Cold-Start Fallbacks & Redis Caching
- **Cold Users**: Detects new users with zero interaction history and provides:
  1. *Popularity Fallback*: Bayesian-smoothed top rated and trending items.
  2. *Content-Based Fallback*: Matches user onboarding genre preferences against item multi-hot genre vectors.
- **Redis Cache**: Formats keys `recsys:rec:{userId}` with configurable TTL (default 3600s). Serves cached recommendations instantly on hits, and populates cache on misses.

### Milestone 7: Triton Serving, Airflow Retraining DAG & Online A/B Testing
- **Triton Serving**: Generates `config.pbtxt` definitions for retrieval and DCN ranker models with dynamic batching and multicore CPU/GPU instance groups.
- **Airflow DAG**: Automated scheduled workflow with 8 topologically ordered tasks (`data_ingestion` $\to$ `spark_features` $\to$ `two_tower` $\to$ `faiss` $\to$ `dcn_ranker` $\to$ `evaluation` $\to$ `triton_deploy` $\to$ `redis_warm`).
- **Online A/B Testing**: Deterministic MD5 hash user bucketing into Control (Baseline Ranker) and Treatment (DCN Ranker). Evaluates NDCG@K, Recall@K, Click-Through Rate (CTR), and two-sample statistical significance ($z$-test on CTR and Welch's $t$-test on NDCG).

---

## Running the Prototype

### Run the Interactive Demo
```bash
python demo.py
```
Outputs a full walkthrough across all 7 milestones: training loss trajectories, FAISS candidate retrieval, DCN re-scoring, cache hit/miss inspection, and the A/B testing scorecard.

### Run Automated Tests
```bash
python -m pytest tests -v
```
Runs the 30-test automated verification suite covering data splitting, Spark features, Two-Tower retrieval, FAISS indexing, DCN ranking, cold-start fallbacks, Redis caching, and A/B test statistics.
