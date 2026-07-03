"""
agents/state.py
---------------
Shared LangGraph state definition.

All three agents (orchestrator classify node, incident extractor,
mitigation generator) read from and write to this TypedDict.

Fields annotated with `Annotated[list, operator.add]` are APPENDED
across nodes instead of overwritten — used for the errors list so
every agent can record its own errors without clobbering others.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    # ── Ingestor-provided ────────────────────────────────────────────────────
    batch_id: str                        # UUID string for this processing batch
    logs: list[dict]                     # all logs in the batch (with severity)
    raw_log_ids: list[int]               # DB ids from raw_logs table

    # ── Orchestrator classify node ────────────────────────────────────────────
    error_logs: list[dict]               # subset of logs with severity ERROR/CRITICAL
    current_status: str                  # pipeline stage tracker

    # ── Incident Extractor node ───────────────────────────────────────────────
    incidents: list[dict]                # structured incidents from LLM
    incident_ids: list[int]              # DB ids from incidents table

    # ── Mitigation Generator node ─────────────────────────────────────────────
    mitigations: list[dict]              # structured mitigations from LLM

    # ── Cross-node accumulator ────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]   # all agent errors, appended
