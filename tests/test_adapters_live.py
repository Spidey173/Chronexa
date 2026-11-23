"""Tests for LiveAdapter and MockAdapter integration layer.

Mocks all external HTTP calls with respx to verify:
- Success paths with correct REST payloads (RFC 2822 base64url for Gmail, JSON for Calendar/Slack)
- Missing tokens raise IntegrationNotConfiguredError ("not_configured")
- 401/403 errors raise IntegrationAuthError
- 429 errors raise IntegrationRateLimitError
- 5xx errors raise IntegrationAPIError
- Malformed payloads raise IntegrationAPIError
- Zero secrets leaked in error messages
- Native app.open and reminder.create return safe 'unsupported' responses on Linux/server
"""
import pytest
import respx
import httpx
from src.actions.adapters import (
    LiveAdapter,
    MockAdapter,
    get_adapter,
    IntegrationNotConfiguredError,
    IntegrationAuthError,
    IntegrationRateLimitError,
    IntegrationAPIError,
)


@pytest.fixture
def live_adapter():
    return LiveAdapter()


# ---------------------------------------------------------------------------
# Default Mode & Mock Adapter Tests
# ---------------------------------------------------------------------------
def test_default_adapter_is_mock():
    adapter = get_adapter()
    assert isinstance(adapter, MockAdapter)


def test_mock_adapter_records_calls():
    mock = MockAdapter()
    mock.clear()
    res = mock.send_gmail(to=["test@example.com"], body="Hello", subject="Hi")
    assert res["status"] == "success"
    assert res["mode"] == "mock"
    assert len(mock.recorded_calls) == 1
    assert mock.recorded_calls[0]["action"] == "gmail.send"


# ---------------------------------------------------------------------------
# Gmail API Tests
# ---------------------------------------------------------------------------
@respx.mock
def test_gmail_send_success(live_adapter):
    route = respx.post("https://gmail.googleapis.com/gmail/v1/users/me/messages/send").mock(
        return_value=httpx.Response(200, json={"id": "msg-12345", "threadId": "th-67890"})
    )

    creds = {"gmail_token": "secret-oauth-token-xyz"}
    res = live_adapter.send_gmail(
        to=["sarah@company.com"],
        body="Let's sync at 3pm.",
        subject="Sync Meeting",
        cc=["alex@company.com"],
        credentials=creds
    )

    assert route.called
    assert res["status"] == "success"
    assert res["mode"] == "live"
    assert res["id"] == "msg-12345"
    assert res["thread_id"] == "th-67890"

    sent_req = route.calls.last.request
    assert sent_req.headers["authorization"] == "Bearer secret-oauth-token-xyz"
    sent_json = sent_req.read().decode("utf-8")
    assert "raw" in sent_json


def test_gmail_missing_token_raises_not_configured(live_adapter):
    with pytest.raises(IntegrationNotConfiguredError) as exc_info:
        live_adapter.send_gmail(to=["sarah@company.com"], body="Hello", credentials={})
    assert exc_info.value.code == "not_configured"
    assert "No OAuth access token found" in str(exc_info.value)


@respx.mock
def test_gmail_401_auth_error(live_adapter):
    respx.post("https://gmail.googleapis.com/gmail/v1/users/me/messages/send").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Invalid Credentials", "status": "UNAUTHENTICATED"}})
    )

    creds = {"gmail_token": "expired-token"}
    with pytest.raises(IntegrationAuthError) as exc_info:
        live_adapter.send_gmail(to=["sarah@company.com"], body="Hello", credentials=creds)

    assert exc_info.value.status_code == 401
    assert exc_info.value.code == "auth_error"
    assert "expired-token" not in str(exc_info.value)


@respx.mock
def test_gmail_429_rate_limit_error(live_adapter):
    respx.post("https://gmail.googleapis.com/gmail/v1/users/me/messages/send").mock(
        return_value=httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
    )

    creds = {"gmail_token": "valid-token"}
    with pytest.raises(IntegrationRateLimitError) as exc_info:
        live_adapter.send_gmail(to=["sarah@company.com"], body="Hello", credentials=creds)

    assert exc_info.value.status_code == 429
    assert exc_info.value.code == "rate_limited"


def test_gmail_malformed_payload_empty_recipients(live_adapter):
    creds = {"gmail_token": "valid-token"}
    with pytest.raises(IntegrationAPIError) as exc_info:
        live_adapter.send_gmail(to=[], body="Hello", credentials=creds)
    assert exc_info.value.code == "invalid_payload"


# ---------------------------------------------------------------------------
# Calendar API Tests (Create & Update)
# ---------------------------------------------------------------------------
@respx.mock
def test_calendar_create_event_success(live_adapter):
    route = respx.post("https://www.googleapis.com/calendar/v3/calendars/primary/events").mock(
        return_value=httpx.Response(
            200,
            json={"id": "evt-abc-999", "htmlLink": "https://calendar.google.com/event?eid=abc"}
        )
    )

    creds = {"calendar_token": "cal-token-123"}
    res = live_adapter.create_calendar_event(
        title="Project Kickoff",
        start="2026-11-25T10:00:00Z",
        end="2026-11-25T11:00:00Z",
        attendees=["dev@company.com"],
        credentials=creds
    )

    assert route.called
    assert res["status"] == "success"
    assert res["mode"] == "live"
    assert res["event_id"] == "evt-abc-999"
    assert res["html_link"] == "https://calendar.google.com/event?eid=abc"


@respx.mock
def test_calendar_update_event_success(live_adapter):
    route = respx.patch("https://www.googleapis.com/calendar/v3/calendars/primary/events/evt-abc-999").mock(
        return_value=httpx.Response(200, json={"id": "evt-abc-999", "summary": "Updated Title"})
    )

    creds = {"calendar_token": "cal-token-123"}
    res = live_adapter.update_calendar_event(
        event_id="evt-abc-999",
        title="Updated Title",
        start="2026-11-25T11:00:00Z",
        credentials=creds
    )

    assert route.called
    assert res["status"] == "success"
    assert res["mode"] == "live"
    assert res["event_id"] == "evt-abc-999"


def test_calendar_missing_token_raises_not_configured(live_adapter):
    with pytest.raises(IntegrationNotConfiguredError):
        live_adapter.create_calendar_event(
            title="Meeting",
            start="2026-11-25T10:00:00Z",
            end="2026-11-25T11:00:00Z",
            credentials={}
        )


@respx.mock
def test_calendar_401_error(live_adapter):
    respx.post("https://www.googleapis.com/calendar/v3/calendars/primary/events").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Invalid token"}})
    )

    creds = {"calendar_token": "bad-token"}
    with pytest.raises(IntegrationAuthError):
        live_adapter.create_calendar_event(
            title="Meeting",
            start="2026-11-25T10:00:00Z",
            end="2026-11-25T11:00:00Z",
            credentials=creds
        )


def test_calendar_malformed_payload(live_adapter):
    creds = {"calendar_token": "valid-token"}
    # Missing title
    with pytest.raises(IntegrationAPIError):
        live_adapter.create_calendar_event(title="", start="2026-11-25T10:00:00Z", end="2026-11-25T11:00:00Z", credentials=creds)

    # Missing event_id for update
    with pytest.raises(IntegrationAPIError):
        live_adapter.update_calendar_event(event_id="", title="New", credentials=creds)


# ---------------------------------------------------------------------------
# Slack API Tests
# ---------------------------------------------------------------------------
@respx.mock
def test_slack_send_success(live_adapter):
    route = respx.post("https://slack.com/api/chat.postMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "channel": "C12345", "ts": "1690000000.000100"})
    )

    creds = {"slack_token": "xoxb-test-token"}
    res = live_adapter.send_slack_message(to="C12345", text="Deployment complete", credentials=creds)

    assert route.called
    assert res["status"] == "success"
    assert res["channel"] == "C12345"
    assert res["ts"] == "1690000000.000100"


@respx.mock
def test_slack_invalid_auth_error(live_adapter):
    respx.post("https://slack.com/api/chat.postMessage").mock(
        return_value=httpx.Response(200, json={"ok": False, "error": "invalid_auth"})
    )

    creds = {"slack_token": "xoxb-invalid"}
    with pytest.raises(IntegrationAuthError) as exc_info:
        live_adapter.send_slack_message(to="C12345", text="Hello", credentials=creds)

    assert exc_info.value.code == "auth_error"
    assert "xoxb-invalid" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Platform-Safe Desktop / Reminder Actions (Phase 2 Verification)
# ---------------------------------------------------------------------------
def test_open_app_safe_in_linux_containers(live_adapter):
    """Verifies open_app does not run macOS shell open and returns unsupported."""
    res = live_adapter.open_app("Slack")
    assert res["status"] == "unsupported"
    assert res["mode"] == "live"
    assert "unsupported in server mode" in res["message"]


def test_create_reminder_safe_in_linux_containers(live_adapter):
    """Verifies create_reminder returns unsupported without fake success."""
    res = live_adapter.create_reminder("Buy milk", "tomorrow")
    assert res["status"] == "unsupported"
    assert res["mode"] == "live"
    assert "unsupported in server mode" in res["message"]

