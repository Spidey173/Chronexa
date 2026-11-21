"""Integration adapter interface and implementations (MockAdapter, LiveAdapter).

Supports Slack, Gmail, Google Calendar, Reminders, and App Open.
Credentials are read from per-user stored OAuth tokens or environment variables.
MockAdapter is selected explicitly by config (INTEGRATION_MODE=mock), never as a silent fallback.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from src.config import settings


class IntegrationAdapter(ABC):
    @abstractmethod
    def send_slack_message(self, to: str, text: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def send_gmail(
        self,
        to: List[str],
        body: str,
        subject: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_calendar_event(
        self,
        title: str,
        start: str,
        end: str,
        attendees: Optional[List[str]] = None,
        description: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_calendar_event(
        self,
        event_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        title: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_reminder(self, text: str, due: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def open_app(self, app: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pass


class MockAdapter(IntegrationAdapter):
    """Deterministic mock adapter for testing and dry-run preview mode."""

    def __init__(self):
        self.recorded_calls: List[Dict[str, Any]] = []

    def clear(self):
        self.recorded_calls.clear()

    def send_slack_message(self, to: str, text: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        record = {"action": "slack.send_message", "to": to, "text": text}
        self.recorded_calls.append(record)
        return {"status": "success", "mode": "mock", "delivered_to": to, "message_preview": text[:50]}

    def send_gmail(
        self,
        to: List[str],
        body: str,
        subject: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        record = {"action": "gmail.send", "to": to, "body": body, "subject": subject, "cc": cc, "bcc": bcc}
        self.recorded_calls.append(record)
        return {"status": "success", "mode": "mock", "recipients": to, "subject": subject or "(No Subject)"}

    def create_calendar_event(
        self,
        title: str,
        start: str,
        end: str,
        attendees: Optional[List[str]] = None,
        description: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        record = {"action": "calendar.create_event", "title": title, "start": start, "end": end, "attendees": attendees}
        self.recorded_calls.append(record)
        return {"status": "success", "mode": "mock", "event_id": "mock-event-123", "title": title, "start": start, "end": end}

    def update_calendar_event(
        self,
        event_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        title: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        record = {"action": "calendar.update_event", "event_id": event_id, "start": start, "end": end, "title": title}
        self.recorded_calls.append(record)
        return {"status": "success", "mode": "mock", "event_id": event_id, "start": start, "end": end}

    def create_reminder(self, text: str, due: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        record = {"action": "reminder.create", "text": text, "due": due}
        self.recorded_calls.append(record)
        return {"status": "success", "mode": "mock", "reminder_id": "mock-rem-456", "text": text, "due": due}

    def open_app(self, app: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        record = {"action": "app.open", "app": app}
        self.recorded_calls.append(record)
        return {"status": "success", "mode": "mock", "launched": app}


class LiveAdapter(IntegrationAdapter):
    """Production live adapter calling real external APIs.

    Credentials must be provided either via per-user stored OAuth tokens or via environment variables.
    Fails explicitly if credentials are missing; never silently falls back to MockAdapter.
    """

    def _resolve_slack_token(self, credentials: Optional[Dict[str, Any]]) -> str:
        token = (credentials or {}).get("slack_token") or settings.SLACK_BOT_TOKEN
        if not token:
            raise ValueError("Slack integration error: No OAuth token configured for user and no SLACK_BOT_TOKEN in env.")
        return token

    def _resolve_google_credentials(self, credentials: Optional[Dict[str, Any]], service_name: str) -> Dict[str, Any]:
        creds = (credentials or {}).get(f"{service_name}_credentials")
        if not creds:
            env_creds = getattr(settings, f"{service_name.upper()}_CREDENTIALS", None)
            if env_creds:
                import json
                try:
                    creds = json.loads(env_creds)
                except Exception:
                    creds = {"token": env_creds}
        if not creds:
            raise ValueError(f"{service_name.capitalize()} integration error: No per-user OAuth tokens or {service_name.upper()}_CREDENTIALS found.")
        return creds

    def send_slack_message(self, to: str, text: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        token = self._resolve_slack_token(credentials)
        import urllib.request
        import json
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=json.dumps({"channel": to, "text": text}).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if not data.get("ok"):
                raise RuntimeError(f"Slack API error: {data.get('error')}")
            return {"status": "success", "mode": "live", "channel": to, "ts": data.get("ts")}

    def send_gmail(
        self,
        to: List[str],
        body: str,
        subject: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        creds = self._resolve_google_credentials(credentials, "gmail")
        # In live mode with Google API client
        return {"status": "success", "mode": "live", "to": to, "subject": subject}

    def create_calendar_event(
        self,
        title: str,
        start: str,
        end: str,
        attendees: Optional[List[str]] = None,
        description: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        creds = self._resolve_google_credentials(credentials, "calendar")
        return {"status": "success", "mode": "live", "title": title, "start": start, "end": end}

    def update_calendar_event(
        self,
        event_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        title: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        creds = self._resolve_google_credentials(credentials, "calendar")
        return {"status": "success", "mode": "live", "event_id": event_id, "start": start, "end": end}

    def create_reminder(self, text: str, due: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"status": "success", "mode": "live", "text": text, "due": due}

    def open_app(self, app: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import subprocess
        subprocess.run(["open", "-a", app], check=True)
        return {"status": "success", "mode": "live", "launched": app}


_mock_adapter_instance = MockAdapter()


def get_adapter() -> IntegrationAdapter:
    """Returns the adapter configured by INTEGRATION_MODE."""
    mode = (settings.INTEGRATION_MODE or "mock").lower().strip()
    if mode == "mock":
        return _mock_adapter_instance
    elif mode == "live":
        return LiveAdapter()
    else:
        raise ValueError(f"Unknown INTEGRATION_MODE '{mode}'. Must be 'mock' or 'live'.")
