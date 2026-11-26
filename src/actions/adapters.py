"""Integration adapter interface and implementations (MockAdapter, LiveAdapter).

Supports Slack, Gmail, Google Calendar, Reminders, and App Open.
Credentials are read from per-user stored OAuth tokens or environment variables.
MockAdapter is selected explicitly by config (INTEGRATION_MODE=mock), never as a silent fallback.
"""
from abc import ABC, abstractmethod
import base64
from email.message import EmailMessage
import json
from typing import Dict, List, Optional, Any
import httpx
from src.core.config import settings


# ---------------------------------------------------------------------------
# Typed Integration Exceptions
# ---------------------------------------------------------------------------
class IntegrationError(Exception):
    """Base exception for external integration errors."""
    def __init__(self, message: str, code: str = "integration_error", status_code: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class IntegrationNotConfiguredError(IntegrationError):
    """Raised when required integration credentials or tokens are missing."""
    def __init__(self, message: str = "Integration credentials not configured"):
        super().__init__(message, code="not_configured", status_code=None)


class IntegrationAuthError(IntegrationError):
    """Raised on 401 Unauthorized or 403 Forbidden responses."""
    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message, code="auth_error", status_code=status_code)


class IntegrationRateLimitError(IntegrationError):
    """Raised on 429 Too Many Requests responses."""
    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message, code="rate_limited", status_code=status_code)


class IntegrationAPIError(IntegrationError):
    """Raised on client (4xx) or server (5xx) API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, code: str = "api_error"):
        super().__init__(message, code=code, status_code=status_code)


# ---------------------------------------------------------------------------
# Adapter Abstract Base Class
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Mock Adapter (Default Dry-Run & Testing Implementation)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Live Adapter (Production Implementation Calling Real REST APIs)
# ---------------------------------------------------------------------------
class LiveAdapter(IntegrationAdapter):
    """Production live adapter calling real external REST APIs.

    Credentials must be provided either via per-user stored OAuth tokens or via environment variables.
    Fails explicitly if credentials are missing; never silently falls back to MockAdapter.
    """

    DEFAULT_TIMEOUT: float = 15.0

    def _resolve_token(self, credentials: Optional[Dict[str, Any]], service_name: str) -> str:
        """Extracts bearer token from credentials dict or environment settings."""
        creds = credentials or {}
        token = (
            creds.get(f"{service_name}_token")
            or creds.get(f"{service_name}_access_token")
            or creds.get("access_token")
            or creds.get("token")
        )

        if not token and f"{service_name}_credentials" in creds:
            nested = creds[f"{service_name}_credentials"]
            if isinstance(nested, dict):
                token = nested.get("access_token") or nested.get("token")
            elif isinstance(nested, str):
                try:
                    parsed = json.loads(nested)
                    if isinstance(parsed, dict):
                        token = parsed.get("access_token") or parsed.get("token")
                except Exception:
                    token = nested

        if not token:
            env_val = getattr(settings, f"{service_name.upper()}_CREDENTIALS", None)
            if env_val:
                try:
                    parsed = json.loads(env_val)
                    if isinstance(parsed, dict):
                        token = parsed.get("access_token") or parsed.get("token")
                    else:
                        token = str(env_val)
                except Exception:
                    token = str(env_val)

        if not token and service_name == "slack":
            token = settings.SLACK_BOT_TOKEN

        if not token and service_name in ("gmail", "calendar"):
            token = getattr(settings, "GOOGLE_ACCESS_TOKEN", None) or getattr(settings, "GOOGLE_CREDENTIALS", None)

        if not token or not str(token).strip():
            raise IntegrationNotConfiguredError(
                f"{service_name.capitalize()} integration error: No OAuth access token found. Expected OAuth token in user credentials or env."
            )
        return str(token).strip()

    def _handle_http_error(self, resp: httpx.Response, service_name: str):
        """Maps HTTP status codes to typed IntegrationError subclasses without logging secrets."""
        status = resp.status_code
        try:
            body_json = resp.json()
            err_detail = body_json.get("error", {})
            if isinstance(err_detail, dict):
                err_msg = err_detail.get("message", resp.text[:200])
            else:
                err_msg = str(err_detail) or resp.text[:200]
        except Exception:
            err_msg = resp.text[:200]

        # Sanitize message - never expose authorization tokens or headers
        err_msg = err_msg.replace("Bearer ", "Bearer [REDACTED]")

        if status in (401, 403):
            raise IntegrationAuthError(
                f"{service_name} API authentication failed (HTTP {status}): {err_msg}",
                status_code=status
            )
        elif status == 429:
            raise IntegrationRateLimitError(
                f"{service_name} API rate limit exceeded (HTTP 429): {err_msg}",
                status_code=429
            )
        elif 400 <= status < 500:
            raise IntegrationAPIError(
                f"{service_name} API client error (HTTP {status}): {err_msg}",
                status_code=status
            )
        else:
            raise IntegrationAPIError(
                f"{service_name} API server error (HTTP {status}): {err_msg}",
                status_code=status
            )

    def _request_json(
        self,
        method: str,
        url: str,
        token: str,
        payload: Dict[str, Any],
        service_name: str,
        content_type: str = "application/json",
    ) -> Dict[str, Any]:
        """Send an authenticated JSON HTTP request and return parsed JSON response."""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        }
        try:
            with httpx.Client(timeout=self.DEFAULT_TIMEOUT) as client:
                resp = client.request(method, url, json=payload, headers=headers)
        except httpx.TimeoutException as e:
            raise IntegrationAPIError(f"{service_name} API timed out: {str(e)}")
        except httpx.RequestError as e:
            raise IntegrationAPIError(f"{service_name} API network error: {str(e)}")

        if not resp.is_success:
            self._handle_http_error(resp, service_name)
        return resp.json()

    def send_slack_message(self, to: str, text: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not to or not to.strip():
            raise IntegrationAPIError("Slack error: target channel/user 'to' must be non-empty", code="invalid_payload")
        if not text or not text.strip():
            raise IntegrationAPIError("Slack error: message 'text' must be non-empty", code="invalid_payload")

        token = self._resolve_token(credentials, "slack")
        data = self._request_json(
            "POST",
            "https://slack.com/api/chat.postMessage",
            token,
            {"channel": to, "text": text},
            "Slack",
            content_type="application/json; charset=utf-8",
        )
        if not data.get("ok"):
            err = data.get("error", "unknown_error")
            if err in ("invalid_auth", "not_authed", "account_inactive", "token_revoked"):
                raise IntegrationAuthError(f"Slack API auth error: {err}", status_code=401)
            elif err == "ratelimited":
                raise IntegrationRateLimitError("Slack API rate limited", status_code=429)
            else:
                raise IntegrationAPIError(f"Slack API error: {err}")

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
        if not to:
            raise IntegrationAPIError("Gmail error: recipient list 'to' must be non-empty", code="invalid_payload")

        token = self._resolve_token(credentials, "gmail")

        msg = EmailMessage()
        msg["To"] = ", ".join(to) if isinstance(to, list) else str(to)
        msg["Subject"] = subject or "(No Subject)"
        if cc:
            msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else str(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc) if isinstance(bcc, list) else str(bcc)
        msg.set_content(body or "")

        raw_bytes = msg.as_bytes()
        raw_b64url = base64.urlsafe_b64encode(raw_bytes).decode("ascii")

        data = self._request_json(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            token,
            {"raw": raw_b64url},
            "Gmail",
        )
        return {
            "status": "success",
            "mode": "live",
            "id": data.get("id"),
            "thread_id": data.get("threadId"),
            "to": to,
            "subject": subject or "(No Subject)"
        }

    def create_calendar_event(
        self,
        title: str,
        start: str,
        end: str,
        attendees: Optional[List[str]] = None,
        description: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not title or not title.strip():
            raise IntegrationAPIError("Calendar error: event 'title' is required", code="invalid_payload")
        if not start or not start.strip():
            raise IntegrationAPIError("Calendar error: event 'start' time is required", code="invalid_payload")
        if not end or not end.strip():
            raise IntegrationAPIError("Calendar error: event 'end' time is required", code="invalid_payload")

        token = self._resolve_token(credentials, "calendar")
        event_body: Dict[str, Any] = {
            "summary": title,
            "start": {"dateTime": start} if "T" in start else {"date": start},
            "end": {"dateTime": end} if "T" in end else {"date": end},
        }
        if description:
            event_body["description"] = description
        if attendees:
            event_body["attendees"] = [{"email": a} for a in attendees]

        data = self._request_json(
            "POST",
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            token,
            event_body,
            "Calendar",
        )
        return {
            "status": "success",
            "mode": "live",
            "event_id": data.get("id"),
            "html_link": data.get("htmlLink"),
            "title": title,
            "start": start,
            "end": end
        }

    def update_calendar_event(
        self,
        event_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        title: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not event_id or not event_id.strip():
            raise IntegrationAPIError("Calendar error: 'event_id' is required for update", code="invalid_payload")

        patch_body: Dict[str, Any] = {}
        if title is not None:
            patch_body["summary"] = title
        if start is not None:
            patch_body["start"] = {"dateTime": start} if "T" in start else {"date": start}
        if end is not None:
            patch_body["end"] = {"dateTime": end} if "T" in end else {"date": end}

        if not patch_body:
            raise IntegrationAPIError("Calendar error: at least one field (title, start, end) must be provided to update", code="invalid_payload")

        token = self._resolve_token(credentials, "calendar")
        data = self._request_json(
            "PATCH",
            f"https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}",
            token,
            patch_body,
            "Calendar",
        )
        return {
            "status": "success",
            "mode": "live",
            "event_id": data.get("id"),
            "title": data.get("summary", title),
            "start": start,
            "end": end
        }

    def create_reminder(self, text: str, due: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Native reminders are unsupported in server/container mode."""
        return {
            "status": "unsupported",
            "mode": "live",
            "action": "reminder.create",
            "text": text,
            "due": due,
            "message": "Native reminders integration is unsupported in server mode."
        }

    def open_app(self, app: str, credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Desktop application launching is unsupported in server/container mode."""
        return {
            "status": "unsupported",
            "mode": "live",
            "action": "app.open",
            "app": app,
            "message": "Desktop application launching is unsupported in server mode."
        }


_mock_adapter_instance = MockAdapter()


def get_adapter() -> IntegrationAdapter:
    """Returns the adapter configured by INTEGRATION_MODE (defaults to MockAdapter)."""
    mode = (settings.INTEGRATION_MODE or "mock").lower().strip()
    if mode == "mock":
        return _mock_adapter_instance
    elif mode == "live":
        return LiveAdapter()
    else:
        raise ValueError(f"Unknown INTEGRATION_MODE '{mode}'. Must be 'mock' or 'live'.")
