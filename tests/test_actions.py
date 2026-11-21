"""Tests for action assistant: strict schema validation, argument limits, rejection of extra args, and offline parsing using pytest.
"""
import pytest
from src.actions.schema import validate_action, ActionValidationError
from src.actions.parser import ActionParser


def test_valid_slack_message():
    act = {
        "type": "slack.send_message",
        "args": {"to": "U03SARAHK", "text": "geocoding looks good"}
    }
    res = validate_action(act)
    assert res["type"] == "slack.send_message"
    assert res["args"]["to"] == "U03SARAHK"


def test_reject_extra_arguments():
    act = {
        "type": "slack.send_message",
        "args": {"to": "U03SARAHK", "text": "looks good", "execute_command": "rm -rf"}
    }
    with pytest.raises(ActionValidationError) as exc_info:
        validate_action(act)
    assert "unauthorized extra arguments" in str(exc_info.value)


def test_reject_unknown_action_type():
    act = {
        "type": "system.run_shell",
        "args": {"cmd": "echo hi"}
    }
    with pytest.raises(ActionValidationError) as exc_info:
        validate_action(act)
    assert "is not allowlisted" in str(exc_info.value)


def test_reject_missing_required_fields():
    act = {
        "type": "gmail.send",
        "args": {"to": ["test@example.com"]}  # missing body
    }
    with pytest.raises(ActionValidationError) as exc_info:
        validate_action(act)
    assert "missing required arguments" in str(exc_info.value)


def test_reject_invalid_email():
    act = {
        "type": "gmail.send",
        "args": {"to": ["not-an-email"], "body": "Hello"}
    }
    with pytest.raises(ActionValidationError) as exc_info:
        validate_action(act)
    assert "invalid email address" in str(exc_info.value)


def test_reject_invalid_datetime_format():
    act = {
        "type": "reminder.create",
        "args": {"text": "Meeting", "due": "tomorrow-maybe"}
    }
    with pytest.raises(ActionValidationError) as exc_info:
        validate_action(act)
    assert "contains invalid datetime" in str(exc_info.value)


def test_calendar_start_must_precede_end():
    act = {
        "type": "calendar.create_event",
        "args": {
            "title": "Discussion",
            "start": "2026-09-17T15:00:00-07:00",
            "end": "2026-09-17T14:00:00-07:00"  # End is before start
        }
    }
    with pytest.raises(ActionValidationError) as exc_info:
        validate_action(act)
    assert "must be after 'start'" in str(exc_info.value)


def test_offline_fallback_destructive_confirmation():
    parser = ActionParser()
    parser.provider = "none"  # Force offline
    actions = parser.parse("Delete all my emails from Marcus", "2026-09-18T09:00:00-07:00")
    assert len(actions) == 1
    assert actions[0]["type"] == "confirm"
    assert "marcus" in actions[0]["args"]["summary"].lower()


def test_offline_fallback_ambiguity_clarification():
    parser = ActionParser()
    parser.provider = "none"
    actions = parser.parse("Message Sarah about the pricing proposal", "2026-09-18T09:00:00-07:00")
    assert len(actions) == 1
    assert actions[0]["type"] == "clarify"
    assert "sarah" in actions[0]["args"]["question"].lower()


def test_offline_fallback_open_app():
    parser = ActionParser()
    parser.provider = "none"
    actions = parser.parse("Open Figma", "2026-09-16T10:00:00-07:00")
    assert len(actions) == 1
    assert actions[0]["type"] == "app.open"
    assert actions[0]["args"]["app"] == "Figma"


def test_offline_fallback_generalized_arbitrary_commands():
    parser = ActionParser()
    parser.provider = "none"

    # 1. Arbitrary app open
    act1 = parser.parse("Open Terminal", "2026-09-16T10:00:00-07:00")
    assert act1[0]["type"] == "app.open"
    assert act1[0]["args"]["app"] == "Terminal"

    # 2. Arbitrary slack message to resolved user
    act2 = parser.parse("Message Dana on Slack saying the design looks fantastic", "2026-09-16T10:00:00-07:00")
    assert act2[0]["type"] == "slack.send_message"
    assert "design looks fantastic" in act2[0]["args"]["text"].lower()

    # 3. Arbitrary email
    act3 = parser.parse("Email Dave about the quarterly review notes", "2026-09-16T10:00:00-07:00")
    assert act3[0]["type"] == "gmail.send"
    assert "dave" in act3[0]["args"]["to"][0].lower()

    # 4. Arbitrary calendar booking
    act4 = parser.parse("Book 45 minutes with Ben tomorrow at 3 about architecture design", "2026-09-16T10:00:00-07:00")
    assert act4[0]["type"] == "calendar.create_event"
    assert "architecture" in act4[0]["args"]["title"].lower()
