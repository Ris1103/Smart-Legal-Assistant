"""MCPClientManager — maintains one ClientSession per MCP server."""
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Holds live SSE sessions to each MCP server."""

    def __init__(self, urls: dict[str, str]):
        self._urls = urls          # {"search": "http://...", ...}
        self._sessions: dict[str, ClientSession] = {}
        self._transports: list[Any] = []

    async def start(self) -> None:
        for name, url in self._urls.items():
            try:
                transport = sse_client(url + "/sse")
                read, write = await transport.__aenter__()
                session = ClientSession(read, write)
                await session.__aenter__()
                await session.initialize()
                self._sessions[name] = session
                self._transports.append((transport, read, write, session))
                logger.info("MCP session started: %s → %s", name, url)
            except Exception as exc:
                logger.warning("Could not connect to MCP server '%s' at %s: %s", name, url, exc)

    async def stop(self) -> None:
        for transport, read, write, session in self._transports:
            try:
                await session.__aexit__(None, None, None)
                await transport.__aexit__(None, None, None)
            except Exception:
                pass
        self._sessions.clear()
        self._transports.clear()

    def get(self, name: str) -> ClientSession | None:
        return self._sessions.get(name)
