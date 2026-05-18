"""
asyncpg connection pool — initialised during FastAPI lifespan.

Usage:
    from db.database import get_db
    async def my_endpoint(conn=Depends(get_db)): ...

When DATABASE_URL is empty (local dev / no Neon account), all DB
operations are skipped gracefully so the rest of the app still works.
"""
import logging
from typing import AsyncGenerator, Optional

import asyncpg
from fastapi import HTTPException

from config.settings import settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def init_pool() -> None:
    global _pool
    if not settings.database_url:
        logger.warning("DATABASE_URL not set — database features disabled.")
        return
    try:
        # asyncpg expects "postgresql://" not "postgresql+asyncpg://"
        dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        _pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
        )
        logger.info("asyncpg pool created (min=%d, max=%d).",
                    settings.db_pool_min_size, settings.db_pool_max_size)
    except Exception as exc:
        logger.error("Failed to create asyncpg pool: %s", exc)
        _pool = None


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        logger.info("asyncpg pool closed.")
        _pool = None


async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """FastAPI dependency — yields a connection from the pool."""
    if _pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Set DATABASE_URL to enable.",
        )
    async with _pool.acquire() as conn:
        yield conn
