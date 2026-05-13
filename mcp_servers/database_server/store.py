"""aiosqlite CRUD for the contracts table."""
import json
import os
import pathlib
from datetime import datetime, timezone

import aiosqlite

_DEFAULT_DB = str(pathlib.Path(__file__).resolve().parent / "contracts.db")
DB_PATH = os.environ.get("CONTRACTS_DB_PATH", _DEFAULT_DB)

_DDL = """
CREATE TABLE IF NOT EXISTS contracts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_type TEXT    NOT NULL,
    params_json   TEXT    NOT NULL,
    rendered_text TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);
"""


async def _db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    await conn.execute(_DDL)
    await conn.commit()
    return conn


async def insert_contract(contract_type: str, params: dict, rendered_text: str) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    async with await _db() as conn:
        cursor = await conn.execute(
            "INSERT INTO contracts (contract_type, params_json, rendered_text, created_at) VALUES (?, ?, ?, ?)",
            (contract_type, json.dumps(params), rendered_text, created_at),
        )
        await conn.commit()
        return {"id": cursor.lastrowid, "created_at": created_at}


async def fetch_contracts(
    contract_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    async with await _db() as conn:
        if contract_type:
            rows = await conn.execute_fetchall(
                "SELECT id, contract_type, params_json, created_at FROM contracts WHERE contract_type=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (contract_type, limit, offset),
            )
        else:
            rows = await conn.execute_fetchall(
                "SELECT id, contract_type, params_json, created_at FROM contracts ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
    return [
        {
            "id": r["id"],
            "contract_type": r["contract_type"],
            "params": json.loads(r["params_json"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


async def fetch_contract(contract_id: int) -> dict:
    async with await _db() as conn:
        rows = await conn.execute_fetchall(
            "SELECT * FROM contracts WHERE id=?", (contract_id,)
        )
    if not rows:
        return {"error": f"Contract {contract_id} not found"}
    r = rows[0]
    return {
        "id": r["id"],
        "contract_type": r["contract_type"],
        "params": json.loads(r["params_json"]),
        "rendered_text": r["rendered_text"],
        "created_at": r["created_at"],
    }
