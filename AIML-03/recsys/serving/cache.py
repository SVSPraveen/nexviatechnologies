"""Redis caching layer for recommendations and retrieval candidates.

Milestone 6: Cache results in Redis with TTL and automatic write-through caching.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import fakeredis

from recsys.config import ServingConfig

logger = logging.getLogger(__name__)

# Check if real redis client is available
try:
    import redis
    _REDIS_CLIENT_AVAILABLE = True
except ImportError:
    _REDIS_CLIENT_AVAILABLE = False


class RecommendationCache:
    """Redis caching layer for recommendation lists and candidate sets."""

    def __init__(self, config: Optional[ServingConfig] = None) -> None:
        self.config = config or ServingConfig()
        self.client: Any = None
        self.is_fake: bool = False
        self._hit_count: int = 0
        self._miss_count: int = 0

        self._init_connection()

    def _init_connection(self) -> None:
        """Attempt to connect to real Redis server; fallback to fakeredis if offline."""
        if _REDIS_CLIENT_AVAILABLE:
            try:
                real_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    socket_connect_timeout=0.5,
                )
                real_client.ping()
                self.client = real_client
                self.is_fake = False
                logger.info("Connected to real Redis server at %s:%d", self.config.redis_host, self.config.redis_port)
                return
            except Exception as exc:
                logger.info("Real Redis server not responding (%s); using in-memory fakeredis instance.", exc)

        self.client = fakeredis.FakeStrictRedis()
        self.is_fake = True
        logger.info("Fakeredis client initialized successfully.")

    def _make_key(self, user_id: int, prefix: str = "rec") -> str:
        return f"recsys:{prefix}:{user_id}"

    def _failover_to_fakeredis(self, error: Exception) -> None:
        """Switch gracefully to in-memory fakeredis upon connection loss."""
        logger.warning("Redis operation failed (%s). Failing over to in-memory fakeredis.", error)
        try:
            self.client = fakeredis.FakeStrictRedis()
            self.is_fake = True
        except Exception as exc:
            logger.error("Failed to initialize fallback fakeredis: %s", exc)

    def get_cached_recommendations(
        self,
        user_id: int,
    ) -> Optional[list[tuple[int, float]]]:
        """Fetch cached recommendations for user_id. Returns None on cache miss."""
        key = self._make_key(user_id, prefix="rec")
        raw = None
        try:
            raw = self.client.get(key)
        except Exception as e:
            self._failover_to_fakeredis(e)
            try:
                raw = self.client.get(key)
            except Exception:
                raw = None

        if raw is None:
            self._miss_count += 1
            return None

        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            self._hit_count += 1
            items = [(int(item["movieId"]), float(item["score"])) for item in payload.get("items", [])]
            return items
        except Exception as e:
            logger.warning("Error parsing cached payload for user %d: %s", user_id, e)
            self._miss_count += 1
            return None

    def cache_recommendations(
        self,
        user_id: int,
        recommendations: list[tuple[int, float]],
        ttl: Optional[int] = None,
    ) -> None:
        """Write recommendations to Redis cache with TTL."""
        key = self._make_key(user_id, prefix="rec")
        expire_time = ttl or self.config.cache_ttl_seconds
        payload = {
            "user_id": int(user_id),
            "timestamp": int(time.time()),
            "items": [{"movieId": int(mid), "score": float(score)} for mid, score in recommendations],
        }
        try:
            self.client.set(key, json.dumps(payload), ex=expire_time)
        except Exception as e:
            self._failover_to_fakeredis(e)
            try:
                self.client.set(key, json.dumps(payload), ex=expire_time)
            except Exception as exc:
                logger.error("Cache write failed after failover: %s", exc)

    def invalidate_user_cache(self, user_id: int) -> None:
        """Evict cached recommendations for a specific user."""
        key = self._make_key(user_id, prefix="rec")
        try:
            self.client.delete(key)
        except Exception as e:
            self._failover_to_fakeredis(e)
            try:
                self.client.delete(key)
            except Exception:
                pass

    def flush_all(self) -> None:
        """Clear all keys in cache database."""
        try:
            self.client.flushdb()
        except Exception as e:
            self._failover_to_fakeredis(e)
            try:
                self.client.flushdb()
            except Exception:
                pass
        self._hit_count = 0
        self._miss_count = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache hit/miss statistics and hit-ratio."""
        total = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total) if total > 0 else 0.0
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "total_requests": total,
            "hit_rate": hit_rate,
            "is_fakeredis": self.is_fake,
        }
