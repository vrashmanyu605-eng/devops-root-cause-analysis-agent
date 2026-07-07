"""
dashboard.py
------------
Simple Gradio dashboard for the DevOps RCA pipeline.

Shows:
  • Live pipeline stats (log counts, incident counts, etc.)
  • Raw Logs viewer with severity filter
  • Incidents table
  • Mitigations viewer (expandable per incident)
  • Agent Runs audit log

Run with:
    python dashboard.py
"""

import json
from datetime import datetime, timezone

import gradio as gr
import psycopg
from dotenv import load_dotenv

load_dotenv()

from config import PG_DB, PG_HOST, PG_PASSWORD, PG_PORT, PG_USER

# =============================================================================
# DB helper  (sync psycopg for Gradio compatibility)
# =============================================================================

def _get_conn():
    return psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
    )


def _ns_to_str(ns: int | None) -> str:
    """Convert nanosecond epoch to a readable UTC string."""
    if ns is None:
        return "—"
    try:
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ns)


def _pg_ts(ts) -> str:
    if ts is None:
        return "—"
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


# =============================================================================
# Data fetchers
# =============================================================================

def fetch_stats():
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM raw_logs")
                total_logs = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM raw_logs WHERE severity IN ('ERROR','CRITICAL')")
                error_logs = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM incidents")
                total_incidents = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM incidents WHERE severity = 'CRITICAL'")
                critical_incidents = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM mitigations")
                total_mitigations = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM agent_runs")
                total_runs = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM agent_runs WHERE status = 'error'")
                failed_runs = cur.fetchone()[0]

                cur.execute("SELECT COUNT(DISTINCT batch_id) FROM raw_logs")
                total_batches = cur.fetchone()[0]

        return (
            f"### 📦 Total Log Lines\n# {total_logs:,}",
            f"### 🔴 Error / Critical Logs\n# {error_logs:,}",
            f"### 🚨 Incidents Detected\n# {total_incidents:,}  ({critical_incidents} CRITICAL)",
            f"### 🛠️ Mitigations Generated\n# {total_mitigations:,}",
            f"### ⚙️ Agent Runs\n# {total_runs:,}  ({failed_runs} failed)",
            f"### 📬 Batches Processed\n# {total_batches:,}",
        )
    except Exception as e:
        msg = f"### ❌ DB Error\n`{e}`"
        return (msg,) * 6


def fetch_raw_logs(severity_filter: str, limit: int):
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                if severity_filter == "ALL":
                    cur.execute(
                        """
                        SELECT id, batch_id, container, host, severity,
                               message, occurrence_count, created_at
                        FROM raw_logs
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, batch_id, container, host, severity,
                               message, occurrence_count, created_at
                        FROM raw_logs
                        WHERE severity = %s
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (severity_filter, limit),
                    )
                rows = cur.fetchall()

        headers = ["ID", "Batch ID", "Container", "Host", "Severity", "Message", "Count", "Created At"]
        data = [
            [r[0], str(r[1])[:8] + "…", r[2], r[3], r[4], r[5][:200], r[6], _pg_ts(r[7])]
            for r in rows
        ]
        return gr.Dataframe(value=data, headers=headers)
    except Exception as e:
        return gr.Dataframe(value=[[str(e)]], headers=["Error"])


def fetch_incidents(severity_filter: str, limit: int):
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                if severity_filter == "ALL":
                    cur.execute(
                        """
                        SELECT id, batch_id, incident_type, severity, container,
                               affected_services, root_cause, error_pattern, log_count, created_at
                        FROM incidents
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, batch_id, incident_type, severity, container,
                               affected_services, root_cause, error_pattern, log_count, created_at
                        FROM incidents
                        WHERE severity = %s
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (severity_filter, limit),
                    )
                rows = cur.fetchall()

        headers = ["ID", "Batch", "Type", "Severity", "Container", "Affected Services", "Root Cause", "Error Pattern", "Log Count", "Created At"]
        data = [
            [
                r[0],
                str(r[1])[:8] + "…",
                r[2] or "—",
                r[3],
                r[4] or "—",
                ", ".join(r[5]) if r[5] else "—",
                (r[6] or "—")[:150],
                (r[7] or "—")[:120],
                r[8],
                _pg_ts(r[9]),
            ]
            for r in rows
        ]
        return gr.Dataframe(value=data, headers=headers)
    except Exception as e:
        return gr.Dataframe(value=[[str(e)]], headers=["Error"])


def fetch_mitigations(limit: int):
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT m.id, m.incident_id, i.incident_type, i.container,
                           m.priority, m.estimated_resolution_time,
                           m.requires_restart, m.commands, m.created_at
                    FROM mitigations m
                    JOIN incidents i ON i.id = m.incident_id
                    ORDER BY m.id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()

        headers = ["ID", "Incident ID", "Incident Type", "Container", "Priority",
                   "Est. Resolution", "Restart?", "Commands", "Created At"]
        data = [
            [
                r[0], r[1], r[2] or "—", r[3] or "—",
                r[4], r[5] or "—",
                "Yes" if r[6] else "No",
                "\n".join(r[7]) if r[7] else "—",
                _pg_ts(r[8]),
            ]
            for r in rows
        ]
        return gr.Dataframe(value=data, headers=headers)
    except Exception as e:
        return gr.Dataframe(value=[[str(e)]], headers=["Error"])


def fetch_mitigation_detail(incident_id_str: str):
    try:
        incident_id = int(incident_id_str.strip())
    except ValueError:
        return "⚠️ Please enter a valid numeric Incident ID."

    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT m.id, m.steps, m.commands, m.prevention_note, m.priority, "
                    "m.estimated_resolution_time, m.requires_restart, mf.feedback, mf.remarks "
                    "FROM mitigations m "
                    "LEFT JOIN mitigation_feedback mf ON mf.mitigation_id = m.id "
                    "WHERE m.incident_id = %s ORDER BY m.id DESC LIMIT 1",
                    (incident_id,),
                )
                row = cur.fetchone()

        if not row:
            return f"No mitigation found for incident ID `{incident_id}`."

        mit_id, steps, commands, prevention, priority, eta, restart, feedback, remarks = row
        steps_parsed = steps if isinstance(steps, list) else json.loads(steps or "[]")

        lines = [
            f"## Mitigation for Incident #{incident_id}",
            f"**Priority:** {priority}   |   **Est. Resolution:** {eta or '—'}   |   **Requires Restart:** {'Yes' if restart else 'No'}",
            "",
            "### 🪜 Steps",
        ]
        for s in steps_parsed:
            lines.append(f"**{s.get('step', '?')}. {s.get('action', '')}**")
            lines.append(f"> {s.get('detail', '')}")
            lines.append("")

        if commands:
            lines += ["### 💻 Commands", "```bash"] + commands + ["```", ""]

        if prevention:
            lines += ["### 🛡️ Prevention", prevention]

        if feedback or remarks:
            lines += ["### 📝 Feedback & Remarks"]
            if feedback:
                lines.append(f"**Feedback:** {feedback}")
            if remarks:
                lines.append(f"**Remarks:** {remarks}")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


def save_feedback_for_mitigation(mitigation_id_str: str, feedback: str, remarks: str):
    try:
        mitigation_id = int(mitigation_id_str.strip())
    except ValueError:
        return "⚠️ Please enter a valid numeric Mitigation ID."

    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mitigation_feedback (mitigation_id, feedback, remarks)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (mitigation_id) DO UPDATE SET
                        feedback = EXCLUDED.feedback,
                        remarks = EXCLUDED.remarks,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (mitigation_id, feedback or None, remarks or None),
                )
                conn.commit()
        return "✅ Feedback saved successfully."
    except Exception as e:
        return f"❌ Error: {e}"


def fetch_agent_runs(agent_filter: str, limit: int):
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                if agent_filter == "ALL":
                    cur.execute(
                        """
                        SELECT id, batch_id, agent_name, status,
                               input_summary, output_summary, error_message,
                               duration_ms, created_at
                        FROM agent_runs
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, batch_id, agent_name, status,
                               input_summary, output_summary, error_message,
                               duration_ms, created_at
                        FROM agent_runs
                        WHERE agent_name = %s
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (agent_filter, limit),
                    )
                rows = cur.fetchall()

        headers = ["ID", "Batch", "Agent", "Status", "Input", "Output", "Error", "Duration (ms)", "Created At"]
        data = [
            [
                r[0],
                str(r[1])[:8] + "…",
                r[2],
                r[3],
                json.dumps(r[4], default=str)[:120] if r[4] else "—",
                json.dumps(r[5], default=str)[:120] if r[5] else "—",
                (r[6] or "—")[:100],
                r[7],
                _pg_ts(r[8]),
            ]
            for r in rows
        ]
        return gr.Dataframe(value=data, headers=headers)
    except Exception as e:
        return gr.Dataframe(value=[[str(e)]], headers=["Error"])


# =============================================================================
# Gradio UI
# =============================================================================

with gr.Blocks(title="DevOps RCA Dashboard", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🔍 DevOps RCA — Live Dashboard
        Real-time view of the agentic log analysis pipeline powered by **LangGraph + Gemini**.
        """
    )

    # ── Stats row ──────────────────────────────────────────────────────────────
    with gr.Row():
        stat_logs      = gr.Markdown("Loading…")
        stat_errors    = gr.Markdown("Loading…")
        stat_incidents = gr.Markdown("Loading…")
    with gr.Row():
        stat_mits      = gr.Markdown("Loading…")
        stat_runs      = gr.Markdown("Loading…")
        stat_batches   = gr.Markdown("Loading…")

    refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="primary")

    def refresh_stats():
        return fetch_stats()

    refresh_stats_btn.click(
        fn=refresh_stats,
        outputs=[stat_logs, stat_errors, stat_incidents, stat_mits, stat_runs, stat_batches],
    )

    demo.load(
        fn=refresh_stats,
        outputs=[stat_logs, stat_errors, stat_incidents, stat_mits, stat_runs, stat_batches],
    )

    gr.Markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1: Raw Logs ───────────────────────────────────────────────────
        with gr.TabItem("📋 Raw Logs"):
            with gr.Row():
                rl_severity = gr.Dropdown(
                    choices=["ALL", "CRITICAL", "ERROR", "WARNING", "INFO"],
                    value="ALL",
                    label="Severity Filter",
                )
                rl_limit = gr.Slider(10, 500, value=50, step=10, label="Max Rows")
            rl_btn = gr.Button("Load Logs", variant="secondary")
            rl_table = gr.Dataframe(interactive=False, wrap=True)
            rl_btn.click(fn=fetch_raw_logs, inputs=[rl_severity, rl_limit], outputs=rl_table)

        # ── Tab 2: Incidents ──────────────────────────────────────────────────
        with gr.TabItem("🚨 Incidents"):
            with gr.Row():
                inc_severity = gr.Dropdown(
                    choices=["ALL", "CRITICAL", "ERROR"],
                    value="ALL",
                    label="Severity Filter",
                )
                inc_limit = gr.Slider(10, 200, value=50, step=10, label="Max Rows")
            inc_btn   = gr.Button("Load Incidents", variant="secondary")
            inc_table = gr.Dataframe(interactive=False, wrap=True)
            inc_btn.click(fn=fetch_incidents, inputs=[inc_severity, inc_limit], outputs=inc_table)

        # ── Tab 3: Mitigations ────────────────────────────────────────────────
        with gr.TabItem("🛠️ Mitigations"):
            mit_limit = gr.Slider(10, 200, value=50, step=10, label="Max Rows")
            mit_btn   = gr.Button("Load Mitigations", variant="secondary")
            mit_table = gr.Dataframe(interactive=False, wrap=True)
            mit_btn.click(fn=fetch_mitigations, inputs=[mit_limit], outputs=mit_table)

            gr.Markdown("#### View Full Mitigation Detail")
            with gr.Row():
                mit_inc_id  = gr.Textbox(label="Incident ID", placeholder="e.g. 3")
                mit_detail_btn = gr.Button("Fetch Detail")
            mit_detail = gr.Markdown()
            mit_detail_btn.click(fn=fetch_mitigation_detail, inputs=[mit_inc_id], outputs=mit_detail)

            gr.Markdown("#### Save Feedback for a Mitigation")
            with gr.Row():
                feedback_mit_id = gr.Textbox(label="Mitigation ID")
                feedback_text = gr.Textbox(label="Feedback", lines=3)
                remarks_text = gr.Textbox(label="Remarks", lines=3)
            feedback_btn = gr.Button("Save Feedback")
            feedback_status = gr.Markdown()
            feedback_btn.click(
                fn=save_feedback_for_mitigation,
                inputs=[feedback_mit_id, feedback_text, remarks_text],
                outputs=feedback_status,
            )

        # ── Tab 4: Agent Runs ─────────────────────────────────────────────────
        with gr.TabItem("⚙️ Agent Runs"):
            with gr.Row():
                ar_agent = gr.Dropdown(
                    choices=["ALL", "orchestrator", "incident_extractor", "mitigation_generator"],
                    value="ALL",
                    label="Agent Filter",
                )
                ar_limit = gr.Slider(10, 500, value=100, step=10, label="Max Rows")
            ar_btn   = gr.Button("Load Agent Runs", variant="secondary")
            ar_table = gr.Dataframe(interactive=False, wrap=True)
            ar_btn.click(fn=fetch_agent_runs, inputs=[ar_agent, ar_limit], outputs=ar_table)

    gr.Markdown(
        "<center><small>DevOps RCA Pipeline — LangGraph · Gemini 2.5 Flash · PostgreSQL</small></center>"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
