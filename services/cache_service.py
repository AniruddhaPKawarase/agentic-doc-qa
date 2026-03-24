"""
Cache Service — L1 in-memory + L2 Redis with semantic query normalization.
──────────────────────────────────────────────────────────────────────────────
L1: cachetools.TTLCache (fast, in-process)
L2: Redis (optional, shared across workers)
Keys: SHA256 of normalized (session_id + query)
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, Optional

from cachetools import TTLCache

logger = logging.getLogger("docqa.cache")


class CacheService:
    """Two-level cache with semantic query normalization."""

    def __init__(
        self,
        l1_maxsize: int = 500,
        l1_ttl: int = 3600,
        redis_url: str = "",
    ):
        self._l1 = TTLCache(maxsize=l1_maxsize, ttl=l1_ttl)
        self._redis = None
        self._redis_url = redis_url

        if redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(redis_url, decode_responses=True)
                logger.info("Redis L2 cache connected")
            except Exception as e:
                logger.warning(f"Redis not available, L2 disabled: {e}")

    def _normalize_query(self, query: str) -> str:
        """Normalize query for semantic matching."""
        q = query.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)  # Remove punctuation
        words = sorted(q.split())  # Sort words
        return " ".join(words)

    def _make_key(self, session_id: str, query: str) -> str:
        """Create cache key from session + normalized query."""
        normalized = self._normalize_query(query)
        raw = f"{session_id}:{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_l1(self, session_id: str, query: str) -> Optional[Dict]:
        """Check L1 cache."""
        key = self._make_key(session_id, query)
        result = self._l1.get(key)
        if result:
            logger.debug(f"L1 cache hit for {key[:12]}...")
        return result

    async def get_l2(self, session_id: str, query: str) -> Optional[Dict]:
        """Check L2 Redis cache."""
        if not self._redis:
            return None
        try:
            key = self._make_key(session_id, query)
            data = await self._redis.get(f"docqa:{key}")
            if data:
                logger.debug(f"L2 cache hit for {key[:12]}...")
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
        return None

    async def get(self, session_id: str, query: str) -> Optional[Dict]:
        """Check L1, then L2."""
        result = self.get_l1(session_id, query)
        if result:
            return result

        result = await self.get_l2(session_id, query)
        if result:
            # Promote to L1
            key = self._make_key(session_id, query)
            self._l1[key] = result
        return result

    def set_l1(self, session_id: str, query: str, value: Dict) -> None:
        """Store in L1 cache."""
        key = self._make_key(session_id, query)
        self._l1[key] = value

    async def set_l2(self, session_id: str, query: str, value: Dict, ttl: int = 3600) -> None:
        """Store in L2 Redis cache."""
        if not self._redis:
            return
        try:
            key = self._make_key(session_id, query)
            await self._redis.setex(f"docqa:{key}", ttl, json.dumps(value))
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    async def set(self, session_id: str, query: str, value: Dict, ttl: int = 3600) -> None:
        """Store in both L1 and L2."""
        self.set_l1(session_id, query, value)
        await self.set_l2(session_id, query, value, ttl)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
