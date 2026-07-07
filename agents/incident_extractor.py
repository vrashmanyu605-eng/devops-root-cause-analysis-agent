"""
agents/incident_extractor.py
-----------------------------
LangGraph node: Incident Extractor Agent

Responsibilities:
  1. Receive a filtered list of ERROR / CRITICAL logs from the orchestrator.
  2. Build a compact prompt and call Gemini (via LiteLLM) to identify distinct
     incidents, root causes, and affected services.
  3. Parse the structured JSON response.
  4. Persist each incident to the `incidents` table.
  5. Return updated state with `incidents` and `incident_ids`.
"""

from __future__ import annotations

import json
import os
import re
import time

import litellm

from agents.state import AgentState
from config import GEMINI_API_KEY, LITELLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from db.queries import insert_incident, log_agent_run
from sentence_transformers import SentenceTransformer
from db.embeddings import save_incident_embedding, build_incident_embedding_text

# ── LiteLLM uses the GEMINI_API_KEY env var for Google models ────────────────
os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a senior DevOps/SRE incident analyst. "
    "You read batches of application and infrastructure logs and identify "
    "distinct incidents or failures. "
    "You ALWAYS respond with valid JSON and NOTHING else."
)

_USER_PROMPT_TEMPLATE = """
Analyse the following log batch and extract every distinct incident or failure.

Return a JSON object with this exact schema:
{{
  "incidents": [
    {{
      "incident_type": "<short category, e.g. DatabaseConnectionFailure | OOMKill | ServiceCrash | NetworkTimeout | DiskFull | AuthFailure>",
      "severity": "<CRITICAL | ERROR>",
      "container": "<container or service name>",
      "affected_services": ["<list of service names involved>"],
      "root_cause": "<one-sentence root cause>",
      "error_pattern": "<the key error string or pattern that identifies this incident>",
      "first_seen": <nanosecond epoch integer of earliest related log, or null>,
      "log_count": <number of log lines that belong to this incident>
    }}
  ]
}}

Rules:
- Group related log lines into a single incident rather than creating one per line.
- If no actionable incident is found, return {{"incidents": []}}.
- Do NOT wrap the JSON in markdown code fences.

LOG BATCH ({count} lines):
{logs}
"""


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response robustly."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"incidents": []}


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------
async def incident_extractor_node(state: AgentState) -> dict:
    """
    LangGraph node: calls Gemini to extract incidents from error_logs.
    Returns a partial state dict (LangGraph merges it into the full state).
    """
    t0 = time.perf_counter()
    batch_id = state["batch_id"]
    error_logs = state.get("error_logs", [])

    if not error_logs:
        return {
            "incidents": [],
            "incident_ids": [],
            "current_status": "no_error_logs",
        }

    # ── Build a compact log payload (cap at 100 entries, truncate long msgs) ──
    compact_logs = [
        {
            "ts": log["timestamp"],
            "container": log.get("container", "unknown"),
            "severity": log.get("severity", "ERROR"),
            "msg": log["message"][:1500],
            "count": log.get("count", 1),
        }
        for log in error_logs[:50]
    ]

    prompt = _USER_PROMPT_TEMPLATE.format(
        count=len(compact_logs),
        logs=json.dumps(compact_logs, indent=2),
    )

    # ── LLM call ─────────────────────────────────────────────────────────────
    try:
        response = await litellm.acompletion(
            model=LITELLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        # response = await litellm.acompletion(
        #     model="openai/gemma-4-e2b",      # name can be almost anything
        #     api_base="http://localhost:1234/v1",
        #     api_key="lm-studio",                     # any non-empty string
        #     messages=[
        #         {"role": "system", "content": _SYSTEM_PROMPT},
        #         {"role": "user", "content": prompt},
        #     ],
        #     temperature=LLM_TEMPERATURE,
        #     max_tokens=LLM_MAX_TOKENS,
        # )
        raw_text: str = response.choices[0].message.content or ""
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        await log_agent_run(
            batch_id=batch_id,
            agent_name="incident_extractor",
            status="error",
            input_summary={"error_log_count": len(error_logs)},
            error_message=str(exc),
            duration_ms=duration_ms,
        )
        print(f"[INCIDENT EXTRACTOR] LLM error: {exc}")
        return {
            "incidents": [],
            "incident_ids": [],
            "current_status": "llm_error",
            "errors": [f"incident_extractor LLM: {exc}"],
        }

    # ── Parse JSON ────────────────────────────────────────────────────────────
    parsed = _extract_json(raw_text)
    incidents: list[dict] = parsed.get("incidents", [])

    # Attach metadata the DB layer needs
    raw_log_ids = state.get("raw_log_ids", [])
    for inc in incidents:
        inc["llm_raw_response"] = raw_text
        inc["raw_log_ids"]      = raw_log_ids

    # ── Embed and persist ──────────────────────────────────────────────────
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # ── Persist to DB ─────────────────────────────────────────────────────────
    incident_ids: list[int] = []
    for inc in incidents:
        try:
            db_id = await insert_incident(inc, batch_id)
            inc["id"] = db_id
            incident_ids.append(db_id)
        except Exception as exc:
            print(f"[INCIDENT EXTRACTOR] DB insert error: {exc}")

    for inc in incidents:
        try:
            embedding_text = build_incident_embedding_text(inc)

            embedding = embedding_model.encode(
                "passage: " + embedding_text,
                normalize_embeddings=True
            ).tolist()

            await save_incident_embedding(
                incident_id=inc["id"],
                embedding=embedding,
                embedded_text=embedding_text,
            )
        except Exception as e:
            print(f"Failed to save embedding: {e}")

    duration_ms = int((time.perf_counter() - t0) * 1000)
    await log_agent_run(
        batch_id=batch_id,
        agent_name="incident_extractor",
        status="success",
        input_summary={"error_log_count": len(error_logs)},
        output_summary={"incident_count": len(incidents)},
        duration_ms=duration_ms,
    )

    print(
        f"[INCIDENT EXTRACTOR] batch={batch_id} "
        f"error_logs={len(error_logs)} incidents={len(incidents)} "
        f"({duration_ms} ms)"
    )

    return {
        "incidents": incidents,
        "incident_ids": incident_ids,
        "current_status": "incidents_extracted",
    }
