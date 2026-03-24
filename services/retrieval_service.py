"""
Retrieval Service — Query embedding → FAISS search → ranked chunks.
──────────────────────────────────────────────────────────────────────────────
Combines embedding + index services into a single retrieval call.
"""

import logging
import time
from typing import List, Tuple

from services.embedding_service import EmbeddingService
from services.index_service import IndexService
from services.file_processor import Chunk
from config import Settings

logger = logging.getLogger("docqa.retrieval")


class RetrievalResult:
    """Container for retrieval output."""

    def __init__(
        self,
        chunks: List[Tuple[Chunk, float]],
        embedding_tokens: int,
        retrieval_ms: float,
    ):
        self.chunks = chunks  # [(chunk, score), ...]
        self.embedding_tokens = embedding_tokens
        self.retrieval_ms = retrieval_ms

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0

    def build_context(self, max_tokens: int = 80000) -> str:
        """Build context string from retrieved chunks, respecting token budget."""
        context_parts = []
        token_count = 0

        for chunk, score in self.chunks:
            if token_count + chunk.token_count > max_tokens:
                break

            source_label = f"[{chunk.file_name}"
            if chunk.page_number:
                source_label += f", page {chunk.page_number}"
            if chunk.sheet_name:
                source_label += f", sheet '{chunk.sheet_name}'"
            source_label += f"] (relevance: {score:.2f})"

            context_parts.append(f"{source_label}\n{chunk.text}")
            token_count += chunk.token_count

        return "\n\n---\n\n".join(context_parts)


class RetrievalService:
    """Orchestrates query embedding + FAISS search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        index_service: IndexService,
        settings: Settings,
    ):
        self.embedding = embedding_service
        self.index = index_service
        self.top_k = settings.retrieval_top_k
        self.score_threshold = settings.retrieval_score_threshold
        self.max_context_tokens = settings.max_context_tokens

    async def retrieve(
        self,
        session_id: str,
        query: str,
    ) -> RetrievalResult:
        """
        Embed query → search FAISS → return ranked chunks.
        """
        start = time.perf_counter()

        # Embed query
        query_vector = await self.embedding.embed_query(query)
        # Estimate embedding tokens (rough: 1 token per ~4 chars)
        embedding_tokens = max(1, len(query) // 4)

        # Search
        results = self.index.search(
            session_id=session_id,
            query_vector=query_vector,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Retrieved {len(results)} chunks for session {session_id} "
            f"in {elapsed_ms:.1f}ms"
        )

        return RetrievalResult(
            chunks=results,
            embedding_tokens=embedding_tokens,
            retrieval_ms=elapsed_ms,
        )
