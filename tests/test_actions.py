"""Tests for action assistant: strict schema validation, argument limits, rejection of extra args, and offline parsing.
"""
import unittest
from datetime import datetime
from src.actions.schema import validate_action, validate_actions_list, ActionValidationError
from src.actions.parser import ActionParser
from src.retrieval.temporal import parse_iso_datetime


class TestActionAssistant(unittest.TestCase):
    def test_valid_slack_message(self):
        act = {
            "type": "slack.send_message",
            "args": {"to": "U03SARAHK", "text": "geocoding looks good"}
        }
        res = validate_action(act)
        self.assertEqual(res["type"], "slack.send_message")
        self.assertEqual(res["args"]["to"], "U03SARAHK")

    def test_reject_extra_arguments(self):
        act = {
            "type": "slack.send_message",
            "args": {"to": "U03SARAHK", "text": "looks good", "execute_command": "rm -rf"}
        }
        with self.assertRaises(ActionValidationError) as cm:
            validate_action(act)
        self.assertIn("unauthorized extra arguments", str(cm.exception))

    def test_reject_unknown_action_type(self):
        act = {
            "type": "system.run_shell",
            "args": {"cmd": "echo hi"}
        }
        with self.assertRaises(ActionValidationError) as cm:
            validate_action(act)
        self.assertIn("is not allowlisted", str(cm.exception))

    def test_reject_missing_required_fields(self):
        act = {
            "type": "gmail.send",
            "args": {"to": ["test@example.com"]}  # missing body
        }
        with self.assertRaises(ActionValidationError) as cm:
            validate_action(act)
        self.assertIn("missing required arguments", str(cm.exception))

    def test_reject_invalid_email(self):
        act = {
            "type": "gmail.send",
            "args": {"to": ["not-an-email"], "body": "Hello"}
        }
        with self.assertRaises(ActionValidationError) as cm:
            validate_action(act)
        self.assertIn("invalid email address", str(cm.exception))

    def test_reject_invalid_datetime_format(self):
        act = {
            "type": "reminder.create",
            "args": {"text": "Meeting", "due": "tomorrow-maybe"}
        }
        with self.assertRaises(ActionValidationError) as cm:
            validate_action(act)
        self.assertIn("contains invalid datetime", str(cm.exception))

    def test_calendar_start_must_precede_end(self):
        act = {
            "type": "calendar.create_event",
            "args": {
                "title": "Discussion",
                "start": "2026-09-17T15:00:00-07:00",
                "end": "2026-09-17T14:00:00-07:00"  # End is before start
            }
        }
        with self.assertRaises(ActionValidationError) as cm:
            validate_action(act)
        self.assertIn("must be after 'start'", str(cm.exception))

    def test_offline_fallback_destructive_confirmation(self):
        parser = ActionParser()
        parser.provider = "none"  # Force offline
        actions = parser.parse("Delete all my emails from Marcus", "2026-09-18T09:00:00-07:00")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["type"], "confirm")
        self.assertIn("marcus", actions[0]["args"]["summary"].lower())

    def test_offline_fallback_ambiguity_clarification(self):
        parser = ActionParser()
        parser.provider = "none"
        actions = parser.parse("Message Sarah about the pricing proposal", "2026-09-18T09:00:00-07:00")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["type"], "clarify")
        self.assertIn("sarah", actions[0]["args"]["question"].lower())

    def test_offline_fallback_open_app(self):
        parser = ActionParser()
        parser.provider = "none"
        actions = parser.parse("Open Figma", "2026-09-16T10:00:00-07:00")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["type"], "app.open")
        self.assertEqual(actions[0]["args"]["app"], "Figma")


if __name__ == "__main__":
    unittest.main()
