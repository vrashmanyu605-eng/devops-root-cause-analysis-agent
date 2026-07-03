"""
db/queries.py
-------------
All SQL helpers used by the agents and the log ingestor.
Every function is async and uses the shared connection pool.
"""

from __future__ import annotations

import json
from typing import Any

from db.connection import get_pool


# =============================================================================
# raw_logs
# =============================================================================

async def insert_raw_logs(logs: list[dict]) -> list[int]:
    """
    Bulk-insert a list of log dicts and return the generated IDs.
    Uses executemany for efficiency on high-volume streams.
    """
    if not logs:
        return []

    pool = await get_pool()
    ids: list[int] = []

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            for log in logs:
                await cur.execute(
                    """
                    INSERT INTO raw_logs
                        (batch_id, timestamp, container, host, severity,
                         message, labels, occurrence_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        log["batch_id"],
                        log["timestamp"],
                        log.get("container", "unknown"),
                        log.get("host", "unknown"),
                        log.get("severity", "INFO"),
                        log["message"],
                        json.dumps(log.get("labels", {})),
                        log.get("count", 1),
                    ),
                )
                row = await cur.fetchone()
                if row:
                    ids.append(row[0])
        await conn.commit()

    return ids


# =============================================================================
# incidents
# =============================================================================

async def insert_incident(incident: dict[str, Any], batch_id: str) -> int:
    """Insert one extracted incident and return its DB id."""
    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO incidents
                    (batch_id, incident_type, severity, container,
                     affected_services, root_cause, error_pattern,
                     first_seen, log_count, raw_log_ids, llm_raw_response)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    batch_id,
                    incident.get("incident_type"),
                    incident.get("severity", "ERROR"),
                    incident.get("container", "unknown"),
                    incident.get("affected_services", []),
                    incident.get("root_cause"),
                    incident.get("error_pattern"),
                    incident.get("first_seen"),
                    incident.get("log_count", 1),
                    incident.get("raw_log_ids", []),
                    incident.get("llm_raw_response"),
                ),
            )
            row = await cur.fetchone()
        await conn.commit()

    return row[0]  # type: ignore[index]


# =============================================================================
# mitigations
# =============================================================================

async def insert_mitigation(
    mitigation: dict[str, Any],
    incident_id: int,
    batch_id: str,
) -> int:
    """Insert one mitigation record and return its DB id."""
    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO mitigations
                    (incident_id, batch_id, priority, steps, commands,
                     estimated_resolution_time, requires_restart,
                     prevention_note, llm_raw_response)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    incident_id,
                    batch_id,
                    mitigation.get("priority", "MEDIUM"),
                    json.dumps(mitigation.get("steps", [])),
                    mitigation.get("commands", []),
                    mitigation.get("estimated_resolution_time"),
                    bool(mitigation.get("requires_restart", False)),
                    mitigation.get("prevention"),
                    mitigation.get("llm_raw_response"),
                ),
            )
            row = await cur.fetchone()
        await conn.commit()

    return row[0]  # type: ignore[index]


# =============================================================================
# agent_runs  (audit trail)
# =============================================================================

async def log_agent_run(
    batch_id: str,
    agent_name: str,
    status: str,
    input_summary: dict | None = None,
    output_summary: dict | None = None,
    error_message: str | None = None,
    duration_ms: int | None = None,
) -> None:
    """Append one row to the agent_runs audit table (fire-and-forget safe)."""
    pool = await get_pool()

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO agent_runs
                    (batch_id, agent_name, status, input_summary,
                     output_summary, error_message, duration_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    batch_id,
                    agent_name,
                    status,
                    json.dumps(input_summary) if input_summary else None,
                    json.dumps(output_summary) if output_summary else None,
                    error_message,
                    duration_ms,
                ),
            )
        await conn.commit()
