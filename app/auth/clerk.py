"""
Clerk JWT verification dependency for FastAPI.

When CLERK_SECRET_KEY is empty (local dev), auth is bypassed and a
mock user payload is returned so all endpoints remain testable without
a Clerk account.
"""
import logging
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, Header
from jose import JWTError, jwt

from config.settings import settings

logger = logging.getLogger(__name__)

_clerk_jwks_cache: Optional[dict] = None


async def _get_clerk_jwks() -> dict:
    """Fetch Clerk's JWKS (cached for the process lifetime)."""
    global _clerk_jwks_cache
    if _clerk_jwks_cache:
        return _clerk_jwks_cache
    # Derive the frontend API URL from the publishable key
    # pk_test_<base64> → https://<base64>.clerk.accounts.dev/.well-known/jwks.json
    pub_key = settings.clerk_publishable_key
    if not pub_key:
        return {}
    try:
        import base64
        # publishable key format: pk_test_<b64encoded-frontend-api>
        b64_part = pub_key.split("_")[-1]
        # pad to valid base64
        padded = b64_part + "=" * (-len(b64_part) % 4)
        frontend_api = base64.b64decode(padded).decode().rstrip("$")
        jwks_url = f"https://{frontend_api}/.well-known/jwks.json"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(jwks_url)
            resp.raise_for_status()
        _clerk_jwks_cache = resp.json()
        return _clerk_jwks_cache
    except Exception as exc:
        logger.warning("Failed to fetch Clerk JWKS: %s", exc)
        return {}


async def require_user(
    authorization: Optional[str] = Header(None),
) -> dict:
    """
    FastAPI dependency — verifies Clerk JWT and returns the payload.

    In dev (CLERK_SECRET_KEY empty): bypasses verification and returns
    a stub payload so endpoints work without a real Clerk account.
    """
    if not settings.clerk_secret_key:
        # Dev mode — no auth
        return {"sub": "dev-user", "email": "dev@local"}

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        jwks = await _get_clerk_jwks()
        # jose can verify against a JWKS dict directly
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        return payload
    except JWTError as exc:
        logger.warning("JWT verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
