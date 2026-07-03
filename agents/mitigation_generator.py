"""
agents/mitigation_generator.py
--------------------------------
LangGraph node: Mitigation Generator Agent

Responsibilities:
  1. Receive the list of incidents produced by the Incident Extractor node.
  2. For each incident, call Gemini (via LiteLLM) to generate a concrete,
     step-by-step mitigation playbook.
  3. Parse the structured JSON response.
  4. Persist each mitigation to the `mitigations` table.
  5. Return updated state with `mitigations`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

import litellm

from agents.state import AgentState
from config import GEMINI_API_KEY, LITELLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
from db.queries import insert_mitigation, log_agent_run

# ── LiteLLM env var for Google models ────────────────────────────────────────
os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a senior DevOps/SRE engineer. "
    "Given an incident description you produce specific, immediately actionable "
    "mitigation steps. "
    "You ALWAYS respond with valid JSON and NOTHING else."
)

_USER_PROMPT_TEMPLATE = """
An incident has been detected in the production environment. Provide a complete mitigation playbook.

INCIDENT:
{incident_json}

Return a JSON object with this exact schema:
{{
  "priority": "<CRITICAL | HIGH | MEDIUM | LOW>",
  "steps": [
    {{
      "step": <integer>,
      "action": "<what to do>",
      "detail": "<specific command, config change, or procedure>"
    }}
  ],
  "commands": ["<ready-to-run shell commands in order>"],
  "estimated_resolution_time": "<e.g. 5-10 minutes>",
  "requires_restart": <true | false>,
  "prevention": "<one-paragraph advice to prevent recurrence>"
}}

Rules:
- Steps must be concrete and specific — no vague advice.
- Commands must be copy-pasteable.
- Do NOT wrap the JSON in markdown code fences.
"""


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response robustly."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}


async def _generate_single(incident: dict, batch_id: str) -> dict | None:
    """Call LLM for one incident; return mitigation dict or None on failure."""
    incident_summary = {
        k: incident.get(k)
        for k in (
            "incident_type", "severity", "container",
            "affected_services", "root_cause", "error_pattern",
        )
    }
    prompt = _USER_PROMPT_TEMPLATE.format(
        incident_json=json.dumps(incident_summary, indent=2)
    )

    # response = await litellm.acompletion(
    #     model=LITELLM_MODEL,
    #     messages=[
    #         {"role": "system", "content": _SYSTEM_PROMPT},
    #         {"role": "user",   "content": prompt},
    #     ],
    #     temperature=0.2,          # slightly higher for creative mitigation ideas
    #     max_tokens=LLM_MAX_TOKENS,
    # )

    response = await litellm.acompletion(
        model="openai/gemma-4-e2b",      # name can be almost anything
        api_base="http://localhost:1234/v1",
        api_key="lm-studio",                     # any non-empty string
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    raw_text: str = response.choices[0].message.content or ""
    mitigation = _extract_json(raw_text)
    mitigation["llm_raw_response"] = raw_text
    return mitigation


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------
async def mitigation_generator_node(state: AgentState) -> dict:
    """
    LangGraph node: calls Gemini concurrently for each incident.
    Returns a partial state dict merged by LangGraph.
    """
    t0 = time.perf_counter()
    batch_id  = state["batch_id"]
    incidents = state.get("incidents", [])

    if not incidents:
        return {"mitigations": [], "current_status": "no_incidents_to_mitigate"}

    # ── Fire LLM calls concurrently for all incidents ─────────────────────────
    tasks = [_generate_single(inc, batch_id) for inc in incidents if inc.get("id")]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_mitigations: list[dict] = []

    for incident, result in zip(
        [inc for inc in incidents if inc.get("id")], results
    ):
        incident_id = incident["id"]

        if isinstance(result, Exception):
            print(f"[MITIGATION GENERATOR] LLM error for incident {incident_id}: {result}")
            continue

        if not result:
            continue

        # ── Persist to DB ─────────────────────────────────────────────────────
        try:
            mit_id = await insert_mitigation(result, incident_id, batch_id)
            result["id"]          = mit_id
            result["incident_id"] = incident_id
            all_mitigations.append(result)
        except Exception as exc:
            print(f"[MITIGATION GENERATOR] DB insert error: {exc}")

    duration_ms = int((time.perf_counter() - t0) * 1000)
    await log_agent_run(
        batch_id=batch_id,
        agent_name="mitigation_generator",
        status="success",
        input_summary={"incident_count": len(incidents)},
        output_summary={"mitigation_count": len(all_mitigations)},
        duration_ms=duration_ms,
    )

    print(
        f"[MITIGATION GENERATOR] batch={batch_id} "
        f"incidents={len(incidents)} mitigations={len(all_mitigations)} "
        f"({duration_ms} ms)"
    )

    return {
        "mitigations": all_mitigations,
        "current_status": "completed",
    }
