"""Dense semantic retrieval component using lightweight ONNX embeddings via FastEmbed.

Features:
- FastEmbed ONNX runtime with CPU acceleration
- Persistent disk caching in .cache/embeddings/ for instant offline repeated runs
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

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT_DIR / ".cache" / "embeddings"


class DenseRetriever:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: Path = CACHE_DIR):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        self.cache_file = self.cache_dir / f"{sanitized_name}.npz"
        
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self._matrix_hashes: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._hash_to_idx: Dict[str, int] = {}
        self._load_disk_cache()
        if HAS_FASTEMBED and TextEmbedding is not None:
            try:
                self.model = TextEmbedding(model_name=model_name)
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
        if self.cache_file.exists():
            try:
                data = np.load(self.cache_file, allow_pickle=True)
                hashes = data["hashes"]
                vectors = data["vectors"]
                for h, vec in zip(hashes, vectors):
                    self.embedding_cache[str(h)] = vec
                self._build_matrix()
                print(f"[DenseRetriever] Loaded {len(hashes)} cached embeddings from {self.cache_file.name}")
            except Exception as e:
                print(f"[DenseRetriever] Warning: Failed to load disk cache ({e}), rebuilding.")

    def _save_disk_cache(self):
        try:
            hashes = np.array(list(self.embedding_cache.keys()))
            vectors = np.array(list(self.embedding_cache.values()))
            np.savez_compressed(self.cache_file, hashes=hashes, vectors=vectors)
            self._build_matrix()
        except Exception as e:
            print(f"[DenseRetriever] Warning: Failed to save disk cache: {e}")

    def is_available(self) -> bool:
        """Returns True if the dense retriever has an active model or cached vectors."""
        return self.model is not None or bool(self.embedding_cache)

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        h = self._text_hash(text)
        if h not in self.embedding_cache:
            if self.model is not None:
                emb = list(self.model.embed([text]))[0]
                norm = np.linalg.norm(emb)
                normed = emb / norm if norm > 0 else emb
                self.embedding_cache[h] = normed
            else:
                return None
        return self.embedding_cache[h]

    def warm_cache(self, units: List[Unit]):
        """Pre-embed units in batches and persist to disk cache."""
        uncached_texts = []
        uncached_hashes = []
        for u in units:
            h = self._text_hash(u.text)
            if h not in self.embedding_cache:
                uncached_texts.append(u.text)
                uncached_hashes.append(h)

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
        for h, emb in zip(unique_hashes, embeddings):
            norm = np.linalg.norm(emb)
            self.embedding_cache[h] = emb / norm if norm > 0 else emb

        self._save_disk_cache()
        print(f"[DenseRetriever] Successfully cached {len(unique_texts)} embeddings.")

    def search(self, query: str, units: List[Unit], top_k: int = 20) -> List[Tuple[Unit, float]]:
        """Rank visible units by cosine similarity against the query.

        Returns an empty list if embeddings cannot be generated or retrieved, preventing
        arbitrary zero-vector ranking noise from corrupting downstream fusion.
        """
        if not units:
            return []

        self.warm_cache(units)
        q_emb = self._get_embedding(query)

        if q_emb is None or not np.any(q_emb):
            return []

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
            emb = self.embedding_cache.get(h)
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
