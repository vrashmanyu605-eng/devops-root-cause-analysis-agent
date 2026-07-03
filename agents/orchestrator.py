"""
agents/orchestrator.py
-----------------------
Builds and exposes the compiled LangGraph StateGraph.

Graph topology:
                         ┌─────────────────────┐
              START ────►│   classify_batch     │  (orchestrator)
                         └────────┬────────────┘
                      error_logs? │
                  ┌───────────────┴──────────────┐
                 YES                              NO
                  │                               │
                  ▼                             END
        ┌──────────────────┐
        │ extract_incidents │  (Incident Extractor)
        └────────┬─────────┘
       incidents? │
   ┌──────────────┴──────────────┐
  YES                            NO
   │                             │
   ▼                           END
┌──────────────────────┐
│ generate_mitigations │  (Mitigation Generator)
└──────────┬───────────┘
           │
          END
"""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from agents.incident_extractor import incident_extractor_node
from agents.mitigation_generator import mitigation_generator_node
from agents.state import AgentState
from db.queries import log_agent_run


# =============================================================================
# Orchestrator classify node
# =============================================================================

async def classify_batch_node(state: AgentState) -> dict:
    """
    Orchestrator node — filters logs by severity and decides routing.
    Logs an audit row to `agent_runs` then returns the error_logs subset.
    """
    t0 = time.perf_counter()
    batch_id = state["batch_id"]
    logs     = state.get("logs", [])

    error_logs = [
        log for log in logs
        if log.get("severity") in ("ERROR", "CRITICAL")
    ]

    severity_summary: dict[str, int] = {}
    for log in logs:
        sev = log.get("severity", "INFO")
        severity_summary[sev] = severity_summary.get(sev, 0) + 1

    route = "extract_incidents" if error_logs else "skip"

    duration_ms = int((time.perf_counter() - t0) * 1000)
    await log_agent_run(
        batch_id=batch_id,
        agent_name="orchestrator",
        status="routing",
        input_summary={"total_logs": len(logs), "severity_breakdown": severity_summary},
        output_summary={"route": route, "error_log_count": len(error_logs)},
        duration_ms=duration_ms,
    )

    print(
        f"[ORCHESTRATOR] batch={batch_id}  total={len(logs)}  "
        + "  ".join(f"{k}={v}" for k, v in severity_summary.items())
        + f"  → {route}"
    )

    return {
        "error_logs":     error_logs,
        "current_status": "classified",
    }


# =============================================================================
# Routing functions (return string keys matched in add_conditional_edges)
# =============================================================================

def _route_after_classify(state: AgentState) -> str:
    return "extract_incidents" if state.get("error_logs") else "end"


def _route_after_extraction(state: AgentState) -> str:
    return "generate_mitigations" if state.get("incidents") else "end"


# =============================================================================
# Graph factory
# =============================================================================

def build_graph():
    """Construct and compile the LangGraph StateGraph. Call once at startup."""
    graph = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    graph.add_node("classify_batch",       classify_batch_node)
    graph.add_node("extract_incidents",    incident_extractor_node)
    graph.add_node("generate_mitigations", mitigation_generator_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("classify_batch")

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "classify_batch",
        _route_after_classify,
        {
            "extract_incidents": "extract_incidents",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "extract_incidents",
        _route_after_extraction,
        {
            "generate_mitigations": "generate_mitigations",
            "end": END,
        },
    )

    graph.add_edge("generate_mitigations", END)

    return graph.compile()
