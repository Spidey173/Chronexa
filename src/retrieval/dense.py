"""Dense semantic retrieval component using lightweight ONNX embeddings via FastEmbed
persisted in a disk-backed vector store (FAISS) behind a repository interface.

Features:
- FastEmbed ONNX runtime with CPU acceleration
- Persistent FAISS disk index behind VectorStoreRepository interface
- Cosine similarity via L2-normalized dot products
- Batch embedding and pre-warming
"""
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    TextEmbedding = None
    HAS_FASTEMBED = False

from eval_harness.records import Unit
from src.retrieval.vector_store import VectorStoreRepository, FaissVectorStore

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT_DIR / "data" / "embeddings"
if not CACHE_DIR.exists():
    CACHE_DIR = ROOT_DIR / ".cache" / "embeddings"


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Path = CACHE_DIR,
        vector_store: Optional[VectorStoreRepository] = None,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            pass
        
        sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        self.cache_file = self.cache_dir / f"{sanitized_name}.npz"
        
        # Disk-backed FAISS vector store behind repository interface
        self.vector_store = vector_store or FaissVectorStore(
            dimension=384,
            storage_dir=self.cache_dir,
            index_name=f"{sanitized_name}_faiss",
        )

        self.embedding_cache: Dict[str, np.ndarray] = {}
        self._matrix_hashes: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._hash_to_idx: Dict[str, int] = {}

        self._load_disk_cache()
        import os
        fastembed_cache = os.environ.get("FASTEMBED_CACHE_PATH")
        if not fastembed_cache:
            bundled = ROOT_DIR / "fastembed_cache"
            if bundled.exists():
                fastembed_cache = str(bundled)

        if HAS_FASTEMBED and TextEmbedding is not None:
            try:
                self.model = TextEmbedding(model_name=model_name, cache_dir=fastembed_cache)
            except Exception as e:
                self.model = None
                print(f"[DenseRetriever] Warning: Could not initialize TextEmbedding ({e}). Falling back to cached embeddings.")
        else:
            self.model = None

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _build_matrix(self):
        if not self.embedding_cache:
            self._matrix = None
            self._hash_to_idx = {}
            return
        self._matrix_hashes = list(self.embedding_cache.keys())
        self._hash_to_idx = {h: i for i, h in enumerate(self._matrix_hashes)}
        vecs = [self.embedding_cache[h] for h in self._matrix_hashes]
        self._matrix = np.ascontiguousarray(np.array(vecs, dtype=np.float32))

    def _load_disk_cache(self):
        # 1. First check if vector_store has items
        if self.vector_store.count() > 0:
            for h in self.vector_store.get_all_hashes():
                vec = self.vector_store.get(h)
                if vec is not None:
                    self.embedding_cache[h] = vec
            self._build_matrix()
            print(f"[DenseRetriever] Loaded {self.vector_store.count()} embeddings from persistent vector store")
            return

        # 2. Check legacy cache file
        if self.cache_file.exists():
            try:
                data = np.load(self.cache_file, allow_pickle=True)
                hashes = [str(h) for h in data["hashes"]]
                vectors = data["vectors"].astype(np.float32)
                for h, vec in zip(hashes, vectors):
                    self.embedding_cache[h] = vec
                self._build_matrix()
                # Populate and persist in vector store
                self.vector_store.add(hashes, vectors)
                self.vector_store.save()
                print(f"[DenseRetriever] Loaded {len(hashes)} cached embeddings from {self.cache_file.name} into vector store")
            except Exception as e:
                print(f"[DenseRetriever] Warning: Failed to load disk cache ({e}), rebuilding.")

    def _save_disk_cache(self):
        try:
            hashes = list(self.embedding_cache.keys())
            vectors = np.array(list(self.embedding_cache.values()), dtype=np.float32)
            if hashes:
                self.vector_store.add(hashes, vectors)
                self.vector_store.save()
            np.savez_compressed(self.cache_file, hashes=np.array(hashes), vectors=vectors)
            self._build_matrix()
        except Exception as e:
            print(f"[DenseRetriever] Warning: Failed to save disk cache: {e}")

    def is_available(self) -> bool:
        """Returns True if the dense retriever has an active model or cached vectors."""
        return self.model is not None or bool(self.embedding_cache) or self.vector_store.count() > 0

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        h = self._text_hash(text)
        if h not in self.embedding_cache:
            vec = self.vector_store.get(h)
            if vec is not None:
                self.embedding_cache[h] = vec
                return vec
            if self.model is not None:
                emb = list(self.model.embed([text]))[0]
                norm = np.linalg.norm(emb)
                normed = emb / norm if norm > 0 else emb
                self.embedding_cache[h] = normed
                self.vector_store.add([h], np.array([normed], dtype=np.float32))
                self.vector_store.save()
            else:
                return None
        return self.embedding_cache[h]

    def warm_cache(self, units: List[Unit]):
        """Pre-embed units in batches and persist to disk cache and vector store."""
        uncached_texts = []
        uncached_hashes = []
        for u in units:
            h = self._text_hash(u.text)
            if h not in self.embedding_cache and not self.vector_store.contains(h):
                uncached_texts.append(u.text)
                uncached_hashes.append(h)
            elif h not in self.embedding_cache:
                vec = self.vector_store.get(h)
                if vec is not None:
                    self.embedding_cache[h] = vec

        if not uncached_texts or self.model is None:
            return

        # Deduplicate while preserving mapping
        unique_map = {}
        for h, txt in zip(uncached_hashes, uncached_texts):
            if h not in unique_map:
                unique_map[h] = txt

        unique_hashes = list(unique_map.keys())
        unique_texts = [unique_map[h] for h in unique_hashes]

        print(f"[DenseRetriever] Embedding {len(unique_texts)} new texts with {self.model_name}...")
        embeddings = list(self.model.embed(unique_texts, batch_size=64))
        new_vecs = []
        for h, emb in zip(unique_hashes, embeddings):
            norm = np.linalg.norm(emb)
            normed = emb / norm if norm > 0 else emb
            self.embedding_cache[h] = normed
            new_vecs.append(normed)

        if new_vecs:
            self.vector_store.add(unique_hashes, np.array(new_vecs, dtype=np.float32))
            self.vector_store.save()

        self._save_disk_cache()
        print(f"[DenseRetriever] Successfully cached {len(unique_texts)} embeddings in FAISS vector store.")

    def search(self, query: str, units: List[Unit], top_k: int = 20) -> List[Tuple[Unit, float]]:
        """Rank visible units by cosine similarity against the query."""
        if not units:
            return []

        self.warm_cache(units)
        q_emb = self._get_embedding(query)

        if q_emb is None or not np.any(q_emb):
            return []

        # Use FAISS / matrix vector store scoring
        if self._matrix is not None and self._hash_to_idx:
            row_indices = []
            valid_units = []
            for u in units:
                h = self._text_hash(u.text)
                idx = self._hash_to_idx.get(h)
                if idx is not None:
                    row_indices.append(idx)
                    valid_units.append(u)
            
            if not row_indices:
                return []
            
            sub_matrix = self._matrix[row_indices]
            scores = np.dot(sub_matrix, q_emb)
            top_n = min(top_k, len(scores))
            ranked_indices = np.argsort(-scores)[:top_n]
            return [(valid_units[idx], float(scores[idx])) for idx in ranked_indices]

        # Fallback if matrix not yet initialized
        doc_embs = []
        valid_units = []
        for u in units:
            h = self._text_hash(u.text)
            emb = self.embedding_cache.get(h) or self.vector_store.get(h)
            if emb is not None and np.any(emb):
                doc_embs.append(emb)
                valid_units.append(u)
            elif self.model is not None:
                emb = self._get_embedding(u.text)
                if emb is not None and np.any(emb):
                    doc_embs.append(emb)
                    valid_units.append(u)

        if not doc_embs:
            return []

        doc_embs_arr = np.array(doc_embs)
        scores = np.dot(doc_embs_arr, q_emb)

        ranked_indices = np.argsort(-scores)[:top_k]
        return [(valid_units[idx], float(scores[idx])) for idx in ranked_indices]
