from db.connection import get_pool


def build_incident_embedding_text(incident: dict) -> str:
    """
    Create a semantic representation of the incident for embedding.
    """

    affected_services = ", ".join(
        incident.get("affected_services", [])
    )

    return f"""
Incident Type: {incident.get("incident_type", "")}
Severity: {incident.get("severity", "")}
Container: {incident.get("container", "")}
Affected Services: {affected_services}
Root Cause: {incident.get("root_cause", "")}
Error Pattern: {incident.get("error_pattern", "")}
""".strip()


async def save_incident_embedding(
    incident_id: int,
    embedding: list[float],
    embedded_text: str,
):
    """
    Save or update an incident embedding.
    """

    pool = await get_pool()

    async with pool.connection() as conn:

        await conn.execute(
            """
            INSERT INTO incident_embeddings
            (
                incident_id,
                embedding,
                embedded_text,
                embedding_model
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s
            )
            ON CONFLICT (incident_id)
            DO UPDATE
            SET
                embedding = EXCLUDED.embedding,
                embedded_text = EXCLUDED.embedded_text,
                embedding_model = EXCLUDED.embedding_model,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                incident_id,
                embedding,
                embedded_text,
                "BAAI/bge-base-en-v1.5",
            ),
        )