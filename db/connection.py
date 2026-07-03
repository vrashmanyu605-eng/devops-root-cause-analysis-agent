"""
db/connection.py
----------------
Async PostgreSQL connection pool using psycopg (v3).
The pool is a singleton — call get_pool() from anywhere; it initialises
once and reuses the same pool for the lifetime of the process.
"""

from __future__ import annotations

import psycopg
from psycopg_pool import AsyncConnectionPool

from config import (
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB,
    PG_POOL_MIN, PG_POOL_MAX,
)

_pool: AsyncConnectionPool | None = None


def _conninfo() -> str:
    return (
        f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} "
        f"user={PG_USER} password={PG_PASSWORD} "
        f"application_name=devops_rca_pipeline"
    )


async def get_pool() -> AsyncConnectionPool:
    """Return the shared async connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=_conninfo(),
            min_size=PG_POOL_MIN,
            max_size=PG_POOL_MAX,
            open=False,
        )
        await _pool.open()
    return _pool


async def close_pool() -> None:
    """Gracefully close the pool on shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
