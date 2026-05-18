"""
Run DDL migrations on startup.  Idempotent — safe to call every boot.
"""
import logging

import asyncpg

logger = logging.getLogger(__name__)

DDL = """
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clerk_id   TEXT UNIQUE NOT NULL,
    email      TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT,
    messages   JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash   TEXT NOT NULL,
    name       TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    last_used  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS contracts (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID REFERENCES users(id) ON DELETE CASCADE,
    contract_type  TEXT NOT NULL,
    params_json    JSONB NOT NULL,
    rendered_text  TEXT NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT now()
);
"""


async def run_migrations(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(DDL)
    logger.info("Database migrations applied.")
