"""
Per-identity rate limiter.

Identity key precedence:
  1. Authenticated JWT user → `user:<uuid>` (a botnet rotating IPs can't bypass)
  2. Client IP             → `ip:<addr>`     (fallback for anonymous traffic)

Storage backend precedence:
  1. Redis fixed-window counters (shared across replicas) when available
  2. In-process buckets (per-instance only) as a graceful fallback

The shared-Redis path is the production default — in-memory only kicks in
during dev when Redis isn't running, so a single replica still gets sane
limits without a hard dependency.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Callable, Optional

from fastapi import Request, Response, HTTPException, status
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.services.cache import cache_service

logger = logging.getLogger(__name__)

# Paths that should never be rate-limited (operational endpoints + static assets)
_EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/", "/metrics"}
_EXEMPT_PREFIXES = ("/uploads/",)


def _extract_user_id(request: Request) -> Optional[str]:
    """Decode the bearer token, return user_id (`sub`), or None if absent/invalid."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    token = auth.split(None, 1)[1].strip()
    if not token or token == "guest":
        return None
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        sub = payload.get("sub")
        return sub if isinstance(sub, str) else None
    except JWTError:
        return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiter scoped per authenticated user (else per IP)."""

    def __init__(
        self,
        app,
        rate_per_minute: Optional[int] = None,
        rate_per_hour: Optional[int] = None,
    ):
        super().__init__(app)
        self.rate_per_minute = rate_per_minute or settings.RATE_LIMIT_PER_MINUTE
        self.rate_per_hour = rate_per_hour or settings.RATE_LIMIT_PER_HOUR
        # Fallback in-process buckets (used only when Redis is unavailable)
        self._minute_buckets: dict[str, list[float]] = defaultdict(list)
        self._hour_buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if path in _EXEMPT_PATHS or any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)

        identity = self._identity_key(request)
        try:
            minute_count, hour_count = await self._consume(identity)
        except HTTPException:
            raise

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rate_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.rate_per_minute - minute_count))
        response.headers["X-RateLimit-Identity"] = identity.split(":", 1)[0]
        return response

    # ── identity ──────────────────────────────────────────────────────────────

    @staticmethod
    def _identity_key(request: Request) -> str:
        user_id = _extract_user_id(request)
        if user_id:
            return f"user:{user_id}"
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    # ── consumption ───────────────────────────────────────────────────────────

    async def _consume(self, identity: str) -> tuple[int, int]:
        """Returns (current minute count, current hour count) AFTER consuming this hit."""
        # Try Redis first — it's the source of truth in production
        if cache_service.redis is not None:
            try:
                return await self._consume_redis(identity)
            except Exception as e:
                # If Redis hiccups, fall through to local — better degraded
                # than 500ing every request.
                logger.warning(f"Redis rate-limit unavailable, falling back to in-process: {e}")

        return self._consume_local(identity)

    async def _consume_redis(self, identity: str) -> tuple[int, int]:
        now = int(time.time())
        minute_window = now // 60
        hour_window = now // 3600
        minute_key = f"rl:m:{identity}:{minute_window}"
        hour_key = f"rl:h:{identity}:{hour_window}"

        # Pipeline so we incr+expire atomically and avoid a round-trip per command
        redis = cache_service.redis
        assert redis is not None
        async with redis.pipeline(transaction=False) as pipe:
            pipe.incr(minute_key, 1)
            pipe.expire(minute_key, 70)  # window + grace
            pipe.incr(hour_key, 1)
            pipe.expire(hour_key, 3700)
            results = await pipe.execute()

        minute_count = int(results[0] or 0)
        hour_count = int(results[2] or 0)

        if minute_count > self.rate_per_minute:
            logger.warning(f"Rate limit exceeded (minute) for {identity}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": str(60 - (now % 60))},
            )
        if hour_count > self.rate_per_hour:
            logger.warning(f"Rate limit exceeded (hour) for {identity}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Hourly rate limit exceeded.",
                headers={"Retry-After": str(3600 - (now % 3600))},
            )
        return minute_count, hour_count

    def _consume_local(self, identity: str) -> tuple[int, int]:
        now = time.time()
        minute = self._minute_buckets[identity] = [t for t in self._minute_buckets[identity] if now - t < 60]
        hour = self._hour_buckets[identity] = [t for t in self._hour_buckets[identity] if now - t < 3600]

        if len(minute) >= self.rate_per_minute:
            logger.warning(f"Rate limit exceeded (minute, local) for {identity}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": "60"},
            )
        if len(hour) >= self.rate_per_hour:
            logger.warning(f"Rate limit exceeded (hour, local) for {identity}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Hourly rate limit exceeded.",
                headers={"Retry-After": "3600"},
            )

        minute.append(now)
        hour.append(now)
        return len(minute), len(hour)
