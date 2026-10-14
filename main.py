"""
main.py
-------
Entry point for the DevOps RCA agentic pipeline.

Starts two concurrent asyncio tasks:
  1. LogIngestor  — polls Loki, processes logs, saves to DB, enqueues batches
  2. Batch processor — dequeues batches and runs them through the LangGraph
                       pipeline (Orchestrator → Incident Extractor → Mitigation Generator)

Usage:
    python main.py
"""

from __future__ import annotations

# ── Windows fix: psycopg3 async requires SelectorEventLoop, not ProactorEventLoop ──
import sys
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import signal

from agents.orchestrator import build_graph
from agents.state import AgentState
from config import QUEUE_MAXSIZE
from db.connection import close_pool, get_pool
from log_ingestor import LogIngestor


# =============================================================================
# Batch consumer
# =============================================================================

async def _process_batches(queue: asyncio.Queue, app) -> None:
    """
    Continuously dequeues log batches and runs them through the compiled
    LangGraph app.  Each batch is fully independent — safe to run multiple
    concurrent consumers if needed (increase NUM_WORKERS in config).
    """
    while True:
        batch = await queue.get()
        try:
            initial_state: AgentState = {
                "batch_id":      batch["batch_id"],
                "logs":          batch["logs"],
                "raw_log_ids":   batch.get("raw_log_ids", []),
                # Filled by agents
                "error_logs":    [],
                "incidents":     [],
                "incident_ids":  [],
                "mitigations":   [],
                "current_status": "pending",
                "errors":        [],
            }
            await app.ainvoke(initial_state)
        except Exception as exc:
            print(f"[PROCESSOR] Error on batch {batch.get('batch_id')}: {exc}")
        finally:
            queue.task_done()


# =============================================================================
# Graceful shutdown helper
# =============================================================================

def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """Set SIGINT / SIGTERM to signal the stop event."""
    loop = asyncio.get_event_loop()

    def _handler():
        print("\n[MAIN] Shutdown signal received …")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler)
        except NotImplementedError:
            # Windows does not support add_signal_handler for all signals
            pass


# =============================================================================
# Main
# =============================================================================

async def _main() -> None:
    _BANNER = r"""
  ____            ___              ____   ____    _
 |  _ \  _____  / _ \ _ __  ___  |  _ \ / ___|  / \
 | | | |/ _ \ \ V / | '_ \/ __| | |_) | |     / _ \
 | |_| |  __/  | |  | |_) \__ \ |  _ <| |___ / ___ \
 |____/ \___|  |_|  | .__/|___/ |_| \_\\____/_/   \_\
                     |_|   Agentic Log Analysis Pipeline
    """
    print(_BANNER)
    print("=" * 65)

    # ── Database ──────────────────────────────────────────────────────────────
    print("[MAIN] Connecting to PostgreSQL …")
    await get_pool()
    print("[MAIN] ✓  Database pool ready")

    # ── LangGraph app ─────────────────────────────────────────────────────────
    print("[MAIN] Compiling LangGraph pipeline …")
    app = build_graph()
    print("[MAIN] ✓  Pipeline compiled")

    # ── Shared queue ──────────────────────────────────────────────────────────
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

    # ── Agents & ingestor ─────────────────────────────────────────────────────
    ingestor      = LogIngestor(queue)
    stop_event    = asyncio.Event()
    _install_signal_handlers(stop_event)

    print("[MAIN] ✓  Starting log ingestor + agent pipeline …\n")
    print("=" * 65)

    ingestor_task  = asyncio.create_task(ingestor.run(),          name="ingestor")
    processor_task = asyncio.create_task(
        _process_batches(queue, app), name="processor"
    )

    # Wait until a shutdown signal arrives or a task crashes
    done, pending = await asyncio.wait(
        [ingestor_task, processor_task, asyncio.create_task(stop_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n[MAIN] Cancelling tasks …")
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    await close_pool()
    print("[MAIN] ✓  Database pool closed.  Bye!")


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        sys.exit(0)
