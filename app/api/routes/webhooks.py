"""
Clerk webhook handler — upserts users into PostgreSQL on Clerk events.

Clerk sends a Svix-signed POST to /webhooks/clerk on user.created /
user.updated. We verify the signature then upsert into the users table.
"""
import json
import logging
from typing import Any, Dict

import asyncpg
from fastapi import APIRouter, Depends, Header, HTTPException, Request

from config.settings import settings
from db.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _verify_svix_signature(
    payload: bytes,
    svix_id: str,
    svix_timestamp: str,
    svix_signature: str,
) -> bool:
    """Verify Svix webhook signature using CLERK_WEBHOOK_SECRET."""
    if not settings.clerk_webhook_secret:
        # Dev mode — skip verification
        return True
    try:
        from svix.webhooks import Webhook
        wh = Webhook(settings.clerk_webhook_secret)
        wh.verify(
            payload,
            {
                "svix-id": svix_id,
                "svix-timestamp": svix_timestamp,
                "svix-signature": svix_signature,
            },
        )
        return True
    except Exception as exc:
        logger.warning("Svix signature verification failed: %s", exc)
        return False


@router.post("/clerk")
async def clerk_webhook(
    request: Request,
    conn: asyncpg.Connection = Depends(get_db),
    svix_id: str = Header(None, alias="svix-id"),
    svix_timestamp: str = Header(None, alias="svix-timestamp"),
    svix_signature: str = Header(None, alias="svix-signature"),
) -> Dict[str, Any]:
    payload = await request.body()
    logger.debug("Clerk webhook received svix_id=%s", svix_id)

    if not _verify_svix_signature(payload, svix_id or "", svix_timestamp or "", svix_signature or ""):
        logger.warning("Clerk webhook signature verification failed svix_id=%s", svix_id)
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    try:
        event = json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.error("Clerk webhook payload is not valid JSON svix_id=%s: %s", svix_id, exc)
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    event_type = event.get("type", "")
    data = event.get("data", {})
    logger.info("Clerk webhook event_type=%s svix_id=%s", event_type, svix_id)

    if event_type in ("user.created", "user.updated"):
        clerk_id = data.get("id")
        email_addresses = data.get("email_addresses", [])
        email = email_addresses[0].get("email_address", "") if email_addresses else ""

        if not clerk_id or not email:
            logger.warning(
                "Clerk webhook %s missing clerk_id or email — skipping upsert",
                event_type,
            )
            return {"status": "ok", "action": "skipped"}

        try:
            await conn.execute(
                """
                INSERT INTO users (clerk_id, email)
                VALUES ($1, $2)
                ON CONFLICT (clerk_id) DO UPDATE SET email = EXCLUDED.email
                """,
                clerk_id,
                email,
            )
            logger.info("Upserted user clerk_id=%s email=%s event=%s", clerk_id, email, event_type)
        except asyncpg.PostgresError as exc:
            logger.error("DB error upserting user clerk_id=%s: %s", clerk_id, exc)
            raise HTTPException(status_code=500, detail="Failed to persist user.")

    return {"status": "ok"}
