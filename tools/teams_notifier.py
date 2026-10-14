import os
import httpx


async def send_teams_notification(
    incident: dict,
    mitigation: dict,
):
    webhook_url = os.getenv("TEAMS_WEBHOOK_URL")

    if not webhook_url:
        print("TEAMS_WEBHOOK_URL not configured")
        return

    # Build mitigation text
    steps = mitigation.get("steps", [])

    if isinstance(steps, list):
        mitigation_text = "\n".join(
            f"{i+1}. {step.get('action','')}"
            for i, step in enumerate(steps)
            if isinstance(step, dict)
        )
    else:
        mitigation_text = mitigation.get("summary", "")

    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.5",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": "🚨 AI Mitigation Generated",
                            "weight": "Bolder",
                            "size": "Large"
                        },
                        {
                            "type": "FactSet",
                            "facts": [
                                {
                                    "title": "Incident",
                                    "value": incident.get("incident_type", "")
                                },
                                {
                                    "title": "Severity",
                                    "value": incident.get("severity", "")
                                },
                                {
                                    "title": "Container",
                                    "value": incident.get("container", "")
                                },
                                {
                                    "title": "Root Cause",
                                    "value": incident.get("root_cause", "")
                                }
                            ]
                        },
                        {
                            "type": "TextBlock",
                            "text": "Mitigation",
                            "weight": "Bolder",
                            "separator": True
                        },
                        {
                            "type": "TextBlock",
                            "text": mitigation_text or "No mitigation generated.",
                            "wrap": True
                        }
                    ]
                }
            }
        ]
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            webhook_url,
            json=payload,
            headers={
                "Content-Type": "application/json"
            }
        )

        print(response.status_code)
        print(response.text)

        response.raise_for_status()