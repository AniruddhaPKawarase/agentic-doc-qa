"""
FAISS Index Service — Per-session vector index management.
──────────────────────────────────────────────────────────────────────────────
Each session gets its own FAISS index for complete data isolation.
Uses IndexFlatIP on L2-normalized vectors (= cosine similarity).
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from pathlib import Path
from services.file_processor import Chunk

logger = logging.getLogger("docqa.index")


@dataclass
class SessionIndex:
    """FAISS index + metadata for a single session."""
    index: faiss.IndexFlatIP
    chunks: List[Chunk] = field(default_factory=list)
    total_vectors: int = 0


class IndexService:
    """Manage per-session FAISS indices."""

    EMBEDDING_DIM = 1536

    def __init__(self):
        self._indices: Dict[str, SessionIndex] = {}
        self._lock = threading.Lock()

    def create_or_update(
        self,
        session_id: str,
        chunks: List[Chunk],
        vectors: np.ndarray,
    ) -> int:
        """
        Add vectors to a session's FAISS index. Creates if doesn't exist.
        Returns total vector count for the session.
        """
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Vector count ({vectors.shape[0]}) != chunk count ({len(chunks)})"
            )

        with self._lock:
            if session_id not in self._indices:
                index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
                self._indices[session_id] = SessionIndex(index=index)

            session_idx = self._indices[session_id]
            session_idx.index.add(vectors)
            session_idx.chunks.extend(chunks)
            session_idx.total_vectors = session_idx.index.ntotal

        # --- S3 SAVE (Phase 7) ---
        self._save_to_s3(session_id, session_idx)
        # --- END S3 SAVE ---

        logger.info(
            f"Session {session_id}: added {len(chunks)} vectors, "
            f"total now {session_idx.total_vectors}"
        )
        return session_idx.total_vectors

    def search(
        self,
        session_id: str,
        query_vector: np.ndarray,
        top_k: int = 8,
        score_threshold: float = 0.30,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search session's FAISS index.
        Returns list of (chunk, score) sorted by descending score.
        Only returns results above score_threshold.
        """
        with self._lock:
            session_idx = self._indices.get(session_id)

        if session_idx is None or session_idx.total_vectors == 0:
            return []

        # Ensure query is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        k = min(top_k, session_idx.total_vectors)
        scores, indices = session_idx.index.search(query_vector, k)

        # Debug: log all raw scores for diagnosis
        valid_scores = [float(scores[0][j]) for j in range(len(indices[0])) if indices[0][j] >= 0]
        logger.info(
            "FAISS scores session=%s: scores=%s threshold=%.2f",
            session_id,
            [f"{s:.4f}" for s in valid_scores[:10]],
            score_threshold,
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            if score < score_threshold:
                continue
            results.append((session_idx.chunks[idx], float(score)))

        return results

    def get_chunk_count(self, session_id: str) -> int:
        """Get total chunk count for a session."""
        with self._lock:
            session_idx = self._indices.get(session_id)
        return session_idx.total_vectors if session_idx else 0

    def get_chunks(self, session_id: str) -> List[Chunk]:
        """Get all chunks for a session."""
        with self._lock:
            session_idx = self._indices.get(session_id)
        return list(session_idx.chunks) if session_idx else []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session's index and free memory."""
        with self._lock:
            if session_id in self._indices:
                del self._indices[session_id]
                logger.info(f"Deleted index for session {session_id}")
                # --- S3 DELETE (Phase 7) ---
                self._delete_from_s3(session_id)
                # --- END S3 DELETE ---
                return True
        return False

    def list_sessions(self) -> Dict[str, int]:
        """Return {session_id: vector_count} for all sessions."""
        with self._lock:
            return {
                sid: si.total_vectors
                for sid, si in self._indices.items()
            }

    # ── S3 Persistence (Phase 7) ──────────────────────────────────────────

    def _save_to_s3(self, session_id: str, session_idx: SessionIndex) -> None:
        """Save FAISS index + chunks to S3."""
        import os, json, tempfile
        if os.getenv("STORAGE_BACKEND", "local") != "s3":
            return
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
            from s3_utils.operations import upload_file, upload_bytes
            from s3_utils.helpers import docqa_session_index_key, docqa_session_chunks_key
            s3_prefix = os.getenv("S3_AGENT_PREFIX", "document-qa-agent")

            # Save FAISS index via temp file
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                faiss.write_index(session_idx.index, tmp.name)
                idx_key = docqa_session_index_key(s3_prefix, session_id)
                upload_file(tmp.name, idx_key)
                os.unlink(tmp.name)

            # Save chunks as JSONL
            chunks_key = docqa_session_chunks_key(s3_prefix, session_id)
            lines = []
            for chunk in session_idx.chunks:
                lines.append(json.dumps({
                    "text": chunk.text,
                    "metadata": chunk.metadata if hasattr(chunk, "metadata") else {},
                }, ensure_ascii=False))
            upload_bytes("\n".join(lines).encode("utf-8"), chunks_key)
        except Exception as e:
            logger.warning(f"S3 index save failed for {session_id}: {e}")

    def _delete_from_s3(self, session_id: str) -> None:
        """Delete session index from S3."""
        import os
        if os.getenv("STORAGE_BACKEND", "local") != "s3":
            return
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
            from s3_utils.operations import delete_prefix
            s3_prefix = os.getenv("S3_AGENT_PREFIX", "document-qa-agent")
            delete_prefix(f"{s3_prefix}/session_data/{session_id}/")
        except Exception as e:
            logger.warning(f"S3 index delete failed for {session_id}: {e}")
