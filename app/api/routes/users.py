"""
User profile and conversation endpoints.
All routes require a valid Clerk JWT (require_user dependency).
"""
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.clerk import require_user
from db.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])
conv_router = APIRouter(prefix="/conversations", tags=["conversations"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_or_create_user(
    conn: asyncpg.Connection, clerk_id: str, email: str
) -> dict:
    row = await conn.fetchrow(
        "SELECT id, clerk_id, email, created_at FROM users WHERE clerk_id = $1",
        clerk_id,
    )
    if row:
        return dict(row)
    row = await conn.fetchrow(
        """
        INSERT INTO users (clerk_id, email)
        VALUES ($1, $2)
        ON CONFLICT (clerk_id) DO UPDATE SET email = EXCLUDED.email
        RETURNING id, clerk_id, email, created_at
        """,
        clerk_id,
        email,
    )
    return dict(row)


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------

@router.get("/me")
async def get_me(
    user: dict = Depends(require_user),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    clerk_id = user["sub"]
    email = user.get("email", "")
    logger.debug("GET /users/me clerk_id=%s", clerk_id)
    try:
        return await _get_or_create_user(conn, clerk_id, email)
    except asyncpg.PostgresError as exc:
        logger.error("DB error fetching user clerk_id=%s: %s", clerk_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch user profile.")


# ---------------------------------------------------------------------------
# Conversation endpoints
# ---------------------------------------------------------------------------

class ConversationCreate(BaseModel):
    title: Optional[str] = None


class MessageAppend(BaseModel):
    role: str   # "user" | "assistant"
    content: str


@conv_router.get("")
async def list_conversations(
    user: dict = Depends(require_user),
    conn: asyncpg.Connection = Depends(get_db),
) -> List[Dict[str, Any]]:
    clerk_id = user["sub"]
    logger.debug("GET /conversations clerk_id=%s", clerk_id)
    try:
        db_user = await _get_or_create_user(conn, clerk_id, user.get("email", ""))
        rows = await conn.fetch(
            """
            SELECT id, title, created_at,
                   jsonb_array_length(messages) AS message_count
            FROM conversations
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            db_user["id"],
        )
        logger.info("Listed %d conversations for clerk_id=%s", len(rows), clerk_id)
        return [dict(r) for r in rows]
    except asyncpg.PostgresError as exc:
        logger.error("DB error listing conversations clerk_id=%s: %s", clerk_id, exc)
        raise HTTPException(status_code=500, detail="Failed to list conversations.")


@conv_router.post("", status_code=201)
async def create_conversation(
    body: ConversationCreate,
    user: dict = Depends(require_user),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    clerk_id = user["sub"]
    logger.debug("POST /conversations clerk_id=%s title=%r", clerk_id, body.title)
    try:
        db_user = await _get_or_create_user(conn, clerk_id, user.get("email", ""))
        row = await conn.fetchrow(
            """
            INSERT INTO conversations (user_id, title)
            VALUES ($1, $2)
            RETURNING id, title, created_at
            """,
            db_user["id"],
            body.title,
        )
        logger.info("Created conversation id=%s clerk_id=%s", row["id"], clerk_id)
        return dict(row)
    except asyncpg.PostgresError as exc:
        logger.error("DB error creating conversation clerk_id=%s: %s", clerk_id, exc)
        raise HTTPException(status_code=500, detail="Failed to create conversation.")


@conv_router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    user: dict = Depends(require_user),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    clerk_id = user["sub"]
    logger.debug("GET /conversations/%s clerk_id=%s", conversation_id, clerk_id)
    try:
        db_user = await _get_or_create_user(conn, clerk_id, user.get("email", ""))
        row = await conn.fetchrow(
            """
            SELECT id, title, messages, created_at
            FROM conversations
            WHERE id = $1 AND user_id = $2
            """,
            conversation_id,
            db_user["id"],
        )
        if not row:
            logger.warning("Conversation %s not found for clerk_id=%s", conversation_id, clerk_id)
            raise HTTPException(status_code=404, detail="Conversation not found.")
        result = dict(row)
        result["messages"] = json.loads(result["messages"])
        return result
    except HTTPException:
        raise
    except asyncpg.PostgresError as exc:
        logger.error("DB error fetching conversation %s clerk_id=%s: %s", conversation_id, clerk_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch conversation.")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse messages for conversation %s: %s", conversation_id, exc)
        raise HTTPException(status_code=500, detail="Conversation data is corrupted.")


@conv_router.post("/{conversation_id}/messages", status_code=201)
async def append_message(
    conversation_id: UUID,
    body: MessageAppend,
    user: dict = Depends(require_user),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    clerk_id = user["sub"]
    logger.debug(
        "POST /conversations/%s/messages clerk_id=%s role=%s",
        conversation_id, clerk_id, body.role,
    )
    try:
        db_user = await _get_or_create_user(conn, clerk_id, user.get("email", ""))
        row = await conn.fetchrow(
            """
            UPDATE conversations
            SET messages = messages || $1::jsonb
            WHERE id = $2 AND user_id = $3
            RETURNING id, messages
            """,
            json.dumps([{"role": body.role, "content": body.content}]),
            conversation_id,
            db_user["id"],
        )
        if not row:
            logger.warning("Conversation %s not found for clerk_id=%s", conversation_id, clerk_id)
            raise HTTPException(status_code=404, detail="Conversation not found.")
        logger.info("Appended message to conversation %s clerk_id=%s", conversation_id, clerk_id)
        return {"id": row["id"], "messages": json.loads(row["messages"])}
    except HTTPException:
        raise
    except asyncpg.PostgresError as exc:
        logger.error("DB error appending message conversation %s clerk_id=%s: %s", conversation_id, clerk_id, exc)
        raise HTTPException(status_code=500, detail="Failed to append message.")
