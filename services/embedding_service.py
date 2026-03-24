"""
Embedding Service — Batch OpenAI embedding with rate limiting.
──────────────────────────────────────────────────────────────────────────────
Embeds chunks using text-embedding-3-small (1536 dims).
Supports batching and concurrency control.
"""

import asyncio
import logging
from typing import List

import numpy as np
from openai import AsyncOpenAI

from config import Settings

logger = logging.getLogger("docqa.embedding")


class EmbeddingService:
    """Async OpenAI embedding with batching."""

    EMBEDDING_DIM = 1536  # text-embedding-3-small

    def __init__(self, settings: Settings):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.batch_size = settings.embedding_batch_size
        self._semaphore = asyncio.Semaphore(3)  # max 3 concurrent API calls

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts. Returns (N, 1536) float32 array.
        Processes in batches with concurrency control.
        """
        if not texts:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        all_embeddings = [None] * len(texts)
        total_tokens = 0

        # Create batch tasks
        tasks = []
        for start in range(0, len(texts), self.batch_size):
            end = min(start + self.batch_size, len(texts))
            batch = texts[start:end]
            tasks.append(self._embed_batch(batch, start, all_embeddings))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Embedding batch failed: {r}")
                raise r
            total_tokens += r

        logger.info(f"Embedded {len(texts)} texts, {total_tokens} tokens total")

        vectors = np.array(all_embeddings, dtype=np.float32)
        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        return vectors

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (1536,) float32 array, L2-normalized."""
        result = await self.embed_texts([query])
        return result[0]

    async def _embed_batch(
        self,
        texts: List[str],
        start_idx: int,
        output: list,
    ) -> int:
        """Embed a single batch, write results into output list. Returns token count."""
        async with self._semaphore:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )

        tokens_used = response.usage.total_tokens if response.usage else 0

        for i, item in enumerate(response.data):
            output[start_idx + i] = item.embedding

        return tokens_used
