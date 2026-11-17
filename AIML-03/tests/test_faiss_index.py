"""Unit tests for Milestone 4: FAISS candidate vector indexing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from recsys.config import FaissConfig
from recsys.retrieval.faiss_index import FaissItemIndex, VectorIndexBackend


def test_vector_index_backend_exact_search() -> None:
    """Verify vector index retrieves correct nearest neighbor under inner product."""
    backend = VectorIndexBackend(dimension=4)
    # 3 vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    backend.add(np.vstack([v1, v2, v3]))

    assert backend.ntotal == 3

    # Query matching v1 exactly
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    scores, indices = backend.search(query, k=2)

    assert indices[0][0] == 0  # v1 is rank 1
    assert indices[0][1] == 2  # v3 is rank 2
    assert np.isclose(scores[0][0], 1.0)
    assert np.isclose(scores[0][1], 0.5)


def test_faiss_item_index_build_and_search() -> None:
    """Verify FaissItemIndex builds and maps candidates accurately."""
    config = FaissConfig(top_k_candidates=3)
    index = FaissItemIndex(embedding_dim=4, config=config)

    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.7, 0.7, 0.0, 0.0],
    ], dtype=np.float32)
    item_ids = [101, 202, 303, 404]

    index.build_index(embeddings, item_ids)
    assert index.total_items == 4

    # Query aligned with item 202
    q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    candidates = index.search_candidates(q, k=2)

    assert len(candidates) == 2
    top_mid, top_score = candidates[0]
    assert top_mid == 202
    assert np.isclose(top_score, 1.0, atol=1e-5)


def test_faiss_index_persistence_save_load() -> None:
    """Verify saving and reloading FAISS index maintains search behavior."""
    index = FaissItemIndex(embedding_dim=3)
    embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mids = [55, 77]
    index.build_index(embs, mids)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = index.save_index(tmp_dir)
        assert (Path(save_path) / "faiss_metadata.npz").exists()

        loaded_index = FaissItemIndex(embedding_dim=3)
        loaded_index.load_index(tmp_dir)

        assert loaded_index.total_items == 2
        assert loaded_index.item_ids == [55, 77]

        # Query loaded index
        q = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        res = loaded_index.search_candidates(q, k=1)
        assert res[0][0] == 77


def test_faiss_empty_index_search() -> None:
    """Verify search on empty FAISS index returns empty list without error."""
    index = FaissItemIndex(embedding_dim=4)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    res = index.search_candidates(q, k=5)
    assert res == []

    # Test k <= 0
    res_zero = index.search_candidates(q, k=0)
    assert res_zero == []

