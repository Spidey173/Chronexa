"""Vector Store repository interface and FAISS disk-backed persistent implementation."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

from src.config import ROOT_DIR

DEFAULT_VECTOR_DIR = ROOT_DIR / ".cache" / "embeddings"


class VectorStoreRepository(ABC):
    """Abstract repository interface for persistent embedding/vector storage."""

    @abstractmethod
    def add(self, hashes: List[str], vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Adds normalized vectors and their content hashes to the store."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Searches the vector store using cosine similarity (dot product of L2-normalized vectors)."""
        pass

    @abstractmethod
    def get(self, hash_str: str) -> Optional[np.ndarray]:
        """Retrieves the vector for a given content hash."""
        pass

    @abstractmethod
    def contains(self, hash_str: str) -> bool:
        """Checks if a hash is stored."""
        pass

    @abstractmethod
    def save(self, path: Optional[Path] = None) -> None:
        """Persists the index to disk."""
        pass

    @abstractmethod
    def load(self, path: Optional[Path] = None) -> bool:
        """Loads the index from disk. Returns True if successfully loaded."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns total number of indexed vectors."""
        pass

    @abstractmethod
    def get_all_hashes(self) -> List[str]:
        """Returns all stored hashes."""
        pass


class FaissVectorStore(VectorStoreRepository):
    """Disk-backed FAISS vector store implementation behind repository interface."""

    def __init__(
        self,
        dimension: int = 384,
        storage_dir: Optional[Path] = None,
        index_name: str = "faiss_index",
    ):
        self.dimension = dimension
        self.storage_dir = Path(storage_dir or DEFAULT_VECTOR_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_name = index_name
        self.index_path = self.storage_dir / f"{index_name}.bin"
        self.meta_path = self.storage_dir / f"{index_name}_meta.json"

        self.idx_to_hash: List[str] = []
        self.hash_to_idx: Dict[str, int] = {}
        self._vectors_cache: Dict[str, np.ndarray] = {}

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None

        # Load from disk if available
        loaded = self.load()
        if not loaded:
            self._try_migrate_legacy_npz()

    def _try_migrate_legacy_npz(self):
        """Migrate legacy BAAI_bge-small-en-v1_5.npz cache into FAISS index if present."""
        legacy_npz = self.storage_dir / "BAAI_bge-small-en-v1_5.npz"
        if legacy_npz.exists():
            try:
                data = np.load(legacy_npz, allow_pickle=True)
                hashes = [str(h) for h in data["hashes"]]
                vectors = data["vectors"].astype(np.float32)
                if len(hashes) == len(vectors) and len(hashes) > 0:
                    self.add(hashes, vectors)
                    self.save()
                    print(f"[FaissVectorStore] Migrated {len(hashes)} legacy embeddings to persistent FAISS index at {self.index_path}")
            except Exception as e:
                print(f"[FaissVectorStore] Legacy migration skipped: {e}")

    def add(self, hashes: List[str], vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if len(hashes) == 0:
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        # Normalize vectors for cosine similarity (dot product)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        new_hashes = []
        new_vecs = []

        for h, v in zip(hashes, vecs):
            if h not in self.hash_to_idx:
                idx = len(self.idx_to_hash)
                self.hash_to_idx[h] = idx
                self.idx_to_hash.append(h)
                self._vectors_cache[h] = v
                new_hashes.append(h)
                new_vecs.append(v)
            else:
                # Update cache
                self._vectors_cache[h] = v

        if new_vecs and HAS_FAISS and self.index is not None:
            self.index.add(np.ascontiguousarray(new_vecs, dtype=np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.count() == 0:
            return []

        q = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        top_k = min(top_k, self.count())

        if HAS_FAISS and self.index is not None and self.index.ntotal > 0:
            scores, indices = self.index.search(q, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.idx_to_hash):
                    results.append((self.idx_to_hash[idx], float(score)))
            return results

        # Fallback in-memory dot product if FAISS is not available
        matrix = np.array([self._vectors_cache[h] for h in self.idx_to_hash], dtype=np.float32)
        dots = np.dot(matrix, q.reshape(-1))
        top_indices = np.argsort(-dots)[:top_k]
        return [(self.idx_to_hash[i], float(dots[i])) for i in top_indices]

    def get(self, hash_str: str) -> Optional[np.ndarray]:
        return self._vectors_cache.get(hash_str)

    def contains(self, hash_str: str) -> bool:
        return hash_str in self.hash_to_idx

    def count(self) -> int:
        return len(self.idx_to_hash)

    def get_all_hashes(self) -> List[str]:
        return list(self.idx_to_hash)

    def save(self, path: Optional[Path] = None) -> None:
        idx_p = path or self.index_path
        meta_p = (path.parent / f"{path.stem}_meta.json") if path else self.meta_path

        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, str(idx_p))

        meta = {
            "dimension": self.dimension,
            "count": len(self.idx_to_hash),
            "idx_to_hash": self.idx_to_hash,
        }
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def load(self, path: Optional[Path] = None) -> bool:
        idx_p = path or self.index_path
        meta_p = (path.parent / f"{path.stem}_meta.json") if path else self.meta_path

        if not idx_p.exists() or not meta_p.exists():
            return False

        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.idx_to_hash = meta.get("idx_to_hash", [])
            self.hash_to_idx = {h: i for i, h in enumerate(self.idx_to_hash)}

            if HAS_FAISS:
                self.index = faiss.read_index(str(idx_p))
                # Reconstruct vectors if index supports it
                if hasattr(self.index, "reconstruct"):
                    for i, h in enumerate(self.idx_to_hash):
                        try:
                            self._vectors_cache[h] = self.index.reconstruct(i)
                        except Exception:
                            pass
            return True
        except Exception as e:
            print(f"[FaissVectorStore] Could not load FAISS index from disk ({e})")
            return False
