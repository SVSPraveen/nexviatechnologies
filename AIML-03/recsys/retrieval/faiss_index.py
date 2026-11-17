"""FAISS candidate indexing and nearest-neighbor retrieval.

Milestone 4: Index item embeddings in FAISS for fast nearest-neighbor search.
Supports IndexFlatIP (exact) and IndexHNSWFlat (approximate nearest neighbor)
with top-K query search and candidate ID mapping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from recsys.config import FaissConfig

logger = logging.getLogger(__name__)

# Check if native FAISS can be loaded without OS Application Control restriction
_FAISS_AVAILABLE = False
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except (ImportError, Exception) as exc:
    logger.info("Native FAISS C-extension unavailable or restricted by OS policy (%s). Using high-performance vector index fallback.", exc)
    _FAISS_AVAILABLE = False


class VectorIndexBackend:
    """Pure NumPy vector index fallback with exact FAISS IndexFlatIP semantics."""

    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.ntotal = 0

    def add(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.vectors.shape[0] == 0:
            self.vectors = x
        else:
            self.vectors = np.vstack([self.vectors, x])
        self.ntotal = self.vectors.shape[0]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute inner product between queries and indexed vectors; return top-k scores and indices."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self.ntotal == 0:
            return np.zeros((len(queries), 0), dtype=np.float32), np.zeros((len(queries), 0), dtype=np.int64)
        
        # Batch inner product: num_queries x ntotal
        sims = np.dot(queries, self.vectors.T)
        k = min(k, self.ntotal)
        
        # Argpartition for top-k selection
        top_k_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        
        # Sort top-k partition descending
        row_indices = np.arange(queries.shape[0])[:, None]
        top_k_sims = sims[row_indices, top_k_idx]
        sorted_order = np.argsort(-top_k_sims, axis=1)
        
        final_idx = np.take_along_axis(top_k_idx, sorted_order, axis=1)
        final_sims = np.take_along_axis(top_k_sims, sorted_order, axis=1)
        return final_sims, final_idx


class FaissItemIndex:
    """FAISS candidate index wrapper supporting HNSW and Flat Inner-Product search."""

    def __init__(
        self,
        embedding_dim: int,
        config: Optional[FaissConfig] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.config = config or FaissConfig()
        self.item_ids: list[int] = []
        self._native_index: Optional[Any] = None
        self._fallback_index: Optional[VectorIndexBackend] = None

        if _FAISS_AVAILABLE:
            try:
                if self.config.index_type.upper() == "HNSW":
                    index = faiss.IndexHNSWFlat(self.embedding_dim, self.config.m)
                    index.hnsw.efConstruction = self.config.ef_construction
                    index.hnsw.efSearch = self.config.ef_search
                    self._native_index = index
                else:
                    self._native_index = faiss.IndexFlatIP(self.embedding_dim)
            except Exception as e:
                logger.warning("Could not initialize native FAISS index: %s. Using vector index.", e)
                self._native_index = None

        if self._native_index is None:
            self._fallback_index = VectorIndexBackend(dimension=self.embedding_dim)

    @property
    def total_items(self) -> int:
        if self._native_index is not None:
            return int(self._native_index.ntotal)
        assert self._fallback_index is not None
        return self._fallback_index.ntotal

    def build_index(
        self,
        embeddings: np.ndarray,
        item_ids: list[int],
    ) -> None:
        """Add item embedding vectors into FAISS index and map ordinal rows to movie IDs."""
        if len(embeddings) != len(item_ids):
            raise ValueError(f"Mismatch between embeddings ({len(embeddings)}) and item_ids ({len(item_ids)})")

        # Normalize vectors for cosine similarity (Inner Product on L2-normalized vectors == Cosine)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = (embeddings / norms).astype(np.float32)

        self.item_ids = list(item_ids)
        if self._native_index is not None:
            self._native_index.reset()
            self._native_index.add(normalized)
        else:
            assert self._fallback_index is not None
            self._fallback_index = VectorIndexBackend(dimension=self.embedding_dim)
            self._fallback_index.add(normalized)

        logger.info("FAISS candidate index built: indexed %d vectors of dim %d.", self.total_items, self.embedding_dim)

    def search_candidates(
        self,
        query_vector: np.ndarray,
        k: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Retrieve top-k nearest candidate items for a given query vector.
        
        Returns:
            list of (movieId, similarity_score) ordered descending by score.
        """
        if self.total_items == 0 or (k is not None and k <= 0):
            return []

        top_k = k or self.config.top_k_candidates
        top_k = min(top_k, self.total_items)
        if top_k <= 0:
            return []

        # Normalize query vector safely
        query_vector = np.nan_to_num(query_vector, nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(query_vector))
        norm = 1.0 if norm == 0.0 else norm
        q = (query_vector / norm).reshape(1, -1).astype(np.float32)

        if self._native_index is not None:
            scores, indices = self._native_index.search(q, top_k)
        else:
            assert self._fallback_index is not None
            scores, indices = self._fallback_index.search(q, top_k)

        results: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.item_ids):
                results.append((self.item_ids[idx], float(score)))

        return results

    def save_index(self, directory: Path | str) -> Path:
        """Persist index and item ID metadata to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        meta_path = target_dir / "faiss_metadata.npz"
        if self._native_index is not None and _FAISS_AVAILABLE:
            faiss.write_index(self._native_index, str(target_dir / "index.faiss"))
            np.savez_compressed(meta_path, item_ids=np.array(self.item_ids), use_native=True)
        else:
            assert self._fallback_index is not None
            np.savez_compressed(
                meta_path,
                item_ids=np.array(self.item_ids),
                vectors=self._fallback_index.vectors,
                use_native=False,
            )
        logger.info("FAISS index saved to %s", target_dir)
        return target_dir

    def load_index(self, directory: Path | str) -> None:
        """Load persisted index and item metadata from disk."""
        target_dir = Path(directory)
        meta_path = target_dir / "faiss_metadata.npz"
        if not meta_path.exists():
            raise FileNotFoundError(f"FAISS metadata not found at {meta_path}")

        with np.load(meta_path) as meta:
            self.item_ids = [int(x) for x in meta["item_ids"]]
            use_native = bool(meta.get("use_native", False))
            vectors = meta["vectors"] if "vectors" in meta else None

        if use_native and _FAISS_AVAILABLE:
            self._native_index = faiss.read_index(str(target_dir / "index.faiss"))
            self._fallback_index = None
        else:
            self._fallback_index = VectorIndexBackend(dimension=self.embedding_dim)
            if vectors is not None:
                self._fallback_index.add(vectors)
            self._native_index = None
        logger.info("FAISS index loaded from %s (%d items)", target_dir, self.total_items)
