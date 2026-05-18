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
    return await _get_or_create_user(conn, clerk_id, email)


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
    db_user = await _get_or_create_user(conn, user["sub"], user.get("email", ""))
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
    return [dict(r) for r in rows]


@conv_router.post("", status_code=201)
async def create_conversation(
    body: ConversationCreate,
    user: dict = Depends(require_user),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    db_user = await _get_or_create_user(conn, user["sub"], user.get("email", ""))
    row = await conn.fetchrow(
        """
        INSERT INTO conversations (user_id, title)
        VALUES ($1, $2)
        RETURNING id, title, created_at
        """,
        db_user["id"],
        body.title,
    )
    return dict(row)


@conv_router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    user: dict = Depends(require_user),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    db_user = await _get_or_create_user(conn, user["sub"], user.get("email", ""))
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
        raise HTTPException(status_code=404, detail="Conversation not found.")
    result = dict(row)
    result["messages"] = json.loads(result["messages"])
    return result


@conv_router.post("/{conversation_id}/messages", status_code=201)
async def append_message(
    conversation_id: UUID,
    body: MessageAppend,
    user: dict = Depends(require_user),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    db_user = await _get_or_create_user(conn, user["sub"], user.get("email", ""))
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
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"id": row["id"], "messages": json.loads(row["messages"])}
