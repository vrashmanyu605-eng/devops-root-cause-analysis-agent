"""
log_ingestor.py
---------------
Async log ingestor — ported from test2.py to asyncio + aiohttp.

Design for high-throughput streams:
  • Uses aiohttp for non-blocking HTTP to Loki
  • Accumulates logs in an in-memory buffer
  • Flushes to the asyncio.Queue (and PostgreSQL) when either:
      – buffer reaches BATCH_SIZE  (volume trigger)
      – BATCH_TIMEOUT seconds have elapsed (time trigger)
  • Queue back-pressure: drops the batch (with a warning) if the
    queue is full, so the ingestor never blocks the event loop.
  • All test2.py logic (multiline merge, deduplication, severity
    detection) is preserved intact.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from typing import Any

import aiohttp

from config import (
    BATCH_SIZE,
    BATCH_TIMEOUT,
    LOKI_LIMIT,
    LOKI_QUERY,
    LOKI_URL,
    POLL_INTERVAL,
)
from db.queries import insert_raw_logs


# =============================================================================
# Severity detection  (unchanged from test2.py)
# =============================================================================

def detect_severity(msg: str) -> str:
    text = msg.lower()
    if any(x in text for x in [
        "panic", "fatal", "outofmemory", "oomkilled", "segmentation fault",
    ]):
        return "CRITICAL"
    if any(x in text for x in [
        "error", "exception", "traceback", "enoent",
        "failed", "refused", "timeout", "login failed",
    ]):
        return "ERROR"
    if any(x in text for x in ["warn", "warning"]):
        return "WARNING"
    return "INFO"


# =============================================================================
# LogIngestor
# =============================================================================

class LogIngestor:
    """
    Continuously polls Loki, processes log lines, and pushes batches onto
    the shared asyncio.Queue for the LangGraph pipeline to consume.
    """

    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue          = queue
        self._last_ts: int | None = None   # nanosecond epoch cursor
        self._buffer: list[dict] = []
        self._last_flush: float = 0.0      # monotonic time of last flush

    # ------------------------------------------------------------------
    # Loki fetch  (async)
    # ------------------------------------------------------------------

    async def _fetch_logs(self, session: aiohttp.ClientSession) -> list[dict]:
        params: dict[str, Any] = {
            "query":     LOKI_QUERY,
            "direction": "forward",
            "limit":     LOKI_LIMIT,
        }
        if self._last_ts is not None:
            params["start"] = str(self._last_ts + 1)

        async with session.get(
            f"{LOKI_URL}/loki/api/v1/query_range",
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        result = data["data"]["result"]
        logs:   list[dict] = []
        newest: int | None = self._last_ts

        for stream in result:
            labels = stream["stream"]
            for ts_str, msg in stream["values"]:
                ts     = int(ts_str)
                newest = ts if newest is None else max(newest, ts)
                logs.append({
                    "timestamp": ts,
                    "container": labels.get("container", "unknown"),
                    "host":      labels.get("host", "unknown"),
                    "labels":    labels,
                    "message":   msg.rstrip(),
                })

        logs.sort(key=lambda x: x["timestamp"])
        self._last_ts = newest
        return logs

    # ------------------------------------------------------------------
    # Processing  (unchanged logic from test2.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_multiline(logs: list[dict]) -> list[dict]:
        merged:  list[dict] = []
        current: dict | None = None

        for log in logs:
            line = log["message"].strip()
            if current is None:
                current = log.copy()
                continue

            if (
                line.startswith("at ")
                or line.startswith("...")
                or line.startswith("errno")
                or line.startswith("code:")
                or line.startswith("syscall")
                or line.startswith("path:")
                or line.startswith("_")
                or line.startswith("}")
                or line.startswith("]")
                or line.startswith(")")
                or line.startswith("{")
                or line.startswith("    ")
            ):
                current["message"] += "\n" + line
            else:
                merged.append(current)
                current = log.copy()

        if current:
            merged.append(current)
        return merged

    @staticmethod
    def _deduplicate(logs: list[dict]) -> list[dict]:
        seen: dict[tuple, dict] = {}
        for log in logs:
            key = (log["container"], log["message"])
            if key not in seen:
                log["count"] = 1
                seen[key]    = log
            else:
                seen[key]["count"] += 1
        return list(seen.values())

    # ------------------------------------------------------------------
    # Flush buffer → DB + queue
    # ------------------------------------------------------------------

    async def _flush(self) -> None:
        if not self._buffer:
            return

        batch    = self._buffer[:]
        self._buffer = []
        batch_id = str(uuid.uuid4())

        # Annotate severity and batch_id
        for log in batch:
            log["severity"] = detect_severity(log["message"])
            log["batch_id"] = batch_id

        # Persist raw logs to PostgreSQL
        try:
            raw_log_ids = await insert_raw_logs(batch)
        except Exception as exc:
            print(f"[INGESTOR] DB insert error: {exc}")
            raw_log_ids = []

        # Log a brief severity summary
        sev_counts: dict[str, int] = defaultdict(int)
        for log in batch:
            sev_counts[log["severity"]] += 1
        print(
            f"[INGESTOR] batch={batch_id}  "
            + "  ".join(f"{k}={v}" for k, v in sev_counts.items())
        )

        # Push to queue (agent pipeline)
        payload = {
            "batch_id":    batch_id,
            "logs":        batch,
            "raw_log_ids": raw_log_ids,
        }
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            print(
                f"[INGESTOR] ⚠  Queue full — dropping batch {batch_id}. "
                "Consider increasing QUEUE_MAXSIZE or reducing BATCH_SIZE."
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._last_flush = asyncio.get_event_loop().time()

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    logs = await self._fetch_logs(session)

                    if logs:
                        logs = self._merge_multiline(logs)
                        logs = self._deduplicate(logs)
                        self._buffer.extend(logs)

                    now = asyncio.get_event_loop().time()
                    time_since_flush = now - self._last_flush

                    if (
                        len(self._buffer) >= BATCH_SIZE
                        or (self._buffer and time_since_flush >= BATCH_TIMEOUT)
                    ):
                        await self._flush()
                        self._last_flush = asyncio.get_event_loop().time()

                except aiohttp.ClientError as exc:
                    print(f"[INGESTOR] Loki connection error: {exc}")
                except Exception as exc:
                    print(f"[INGESTOR] Unexpected error: {exc}")

                await asyncio.sleep(POLL_INTERVAL)
