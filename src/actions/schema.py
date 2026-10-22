"""Strict schema definitions and validation for Chronexa Action Assistant (TextOS).

Enforces strict schemas for all dry-run actions:
- Allowlisted action types
- Required arguments and expected types
- Explicit rejection of extra, unknown, or malformed arguments
- Sensible length and format bounds
- Datetime ISO 8601 parsing and validation
"""
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Allowlisted action types
ALLOWLISTED_ACTION_TYPES: Set[str] = {
    "slack.send_message",
    "gmail.send",
    "calendar.create_event",
    "calendar.update_event",
    "reminder.create",
    "memory.ask",
    "app.open",
    "clarify",
    "confirm"
}

# Strict argument specifications
# (required_keys, optional_keys)
ACTION_SPECS: Dict[str, Dict[str, Set[str]]] = {
    "slack.send_message": {
        "required": {"to", "text"},
        "optional": set()
    },
    "gmail.send": {
        "required": {"to", "body"},
        "optional": {"subject", "cc"}
    },
    "calendar.create_event": {
        "required": {"title", "start", "end"},
        "optional": {"attendees", "location", "description"}
    },
    "calendar.update_event": {
        "required": {"event_id"},
        "optional": {"start", "end", "title", "attendees", "location", "description", "status"}
    },
    "reminder.create": {
        "required": {"text", "due"},
        "optional": set()
    },
    "memory.ask": {
        "required": {"question"},
        "optional": set()
    },
    "app.open": {
        "required": {"app"},
        "optional": set()
    },
    "clarify": {
        "required": {"question"},
        "optional": set()
    },
    "confirm": {
        "required": {"summary"},
        "optional": set()
    }
}


class ActionValidationError(ValueError):
    """Raised when an action dictionary violates schema, type, or constraint rules."""
    pass


def _validate_iso_datetime(value: Any, field_name: str) -> datetime:
    """Validates that a value is a valid ISO 8601 datetime string."""
    if not isinstance(value, str):
        raise ActionValidationError(f"Field '{field_name}' must be an ISO 8601 string, got {type(value).__name__}")
    s = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            # Default naive datetimes to Alex's local workspace timezone
            dt = dt.replace(tzinfo=LOCAL_TZ)
        return dt
    except Exception as e:
        raise ActionValidationError(f"Field '{field_name}' contains invalid datetime '{value}': {e}")


def _validate_email(email: str, field_name: str):
    if not isinstance(email, str) or not EMAIL_REGEX.match(email.strip()):
        raise ActionValidationError(f"Field '{field_name}' contains invalid email address '{email}'")


def validate_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Validates an action against strict schema definitions.
    
    Returns the sanitized, validated action dictionary.
    Raises ActionValidationError if the action is malformed, has unknown types,
    is missing required fields, or has extra unknown arguments.
    """
    if not isinstance(action, dict):
        raise ActionValidationError(f"Action must be a dictionary, got {type(action).__name__}")

    # Check top-level keys
    allowed_top_keys = {"type", "args"}
    extra_top_keys = set(action.keys()) - allowed_top_keys
    if extra_top_keys:
        raise ActionValidationError(f"Action has unexpected top-level keys: {sorted(list(extra_top_keys))}")

    act_type = action.get("type")
    if not act_type or not isinstance(act_type, str):
        raise ActionValidationError("Action missing required 'type' string")

    if act_type not in ALLOWLISTED_ACTION_TYPES:
        raise ActionValidationError(f"Action type '{act_type}' is not allowlisted. Allowed: {sorted(list(ALLOWLISTED_ACTION_TYPES))}")

    args = action.get("args")
    if not isinstance(args, dict):
        raise ActionValidationError(f"Action 'args' must be a dictionary, got {type(args).__name__}")

    spec = ACTION_SPECS[act_type]
    req_keys = spec["required"]
    opt_keys = spec["optional"]
    all_allowed_keys = req_keys | opt_keys

    # Check for missing required keys
    missing_keys = req_keys - set(args.keys())
    if missing_keys:
        raise ActionValidationError(f"Action '{act_type}' missing required arguments: {sorted(list(missing_keys))}")

    # Check for extra/unknown arguments (strict rejection)
    extra_keys = set(args.keys()) - all_allowed_keys
    if extra_keys:
        raise ActionValidationError(f"Action '{act_type}' contains unauthorized extra arguments: {sorted(list(extra_keys))}")

    # Type & constraint checks per action
    validated_args = {}

    if act_type == "slack.send_message":
        to_val = args["to"]
        if not isinstance(to_val, str) or not to_val.strip() or len(to_val) > 120:
            raise ActionValidationError("Field 'to' in slack.send_message must be a non-empty string <= 120 chars")
        text_val = args["text"]
        if not isinstance(text_val, str) or not text_val.strip() or len(text_val) > 10000:
            raise ActionValidationError("Field 'text' in slack.send_message must be a non-empty string <= 10000 chars")
        validated_args["to"] = to_val.strip()
        validated_args["text"] = text_val.strip()

    elif act_type == "gmail.send":
        to_raw = args["to"]
        to_list = [to_raw] if isinstance(to_raw, str) else to_raw
        if not isinstance(to_list, list) or not to_list:
            raise ActionValidationError("Field 'to' in gmail.send must be a non-empty email list")
        for em in to_list:
            _validate_email(em, "to")
        body_val = args["body"]
        if not isinstance(body_val, str) or not body_val.strip() or len(body_val) > 20000:
            raise ActionValidationError("Field 'body' in gmail.send must be a non-empty string <= 20000 chars")
        validated_args["to"] = [e.strip() for e in to_list]
        validated_args["body"] = body_val.strip()
        if "subject" in args:
            subj = args["subject"]
            if not isinstance(subj, str) or len(subj) > 500:
                raise ActionValidationError("Field 'subject' in gmail.send must be a string <= 500 chars")
            validated_args["subject"] = subj.strip()
        if "cc" in args:
            cc_raw = args["cc"]
            cc_list = [cc_raw] if isinstance(cc_raw, str) else cc_raw
            if not isinstance(cc_list, list):
                raise ActionValidationError("Field 'cc' in gmail.send must be a list of emails")
            for em in cc_list:
                _validate_email(em, "cc")
            validated_args["cc"] = [e.strip() for e in cc_list]

    elif act_type == "calendar.create_event":
        title = args["title"]
        if not isinstance(title, str) or not title.strip() or len(title) > 300:
            raise ActionValidationError("Field 'title' in calendar.create_event must be a string <= 300 chars")
        st_dt = _validate_iso_datetime(args["start"], "start")
        en_dt = _validate_iso_datetime(args["end"], "end")
        if en_dt <= st_dt:
            raise ActionValidationError(f"calendar.create_event 'end' ({args['end']}) must be after 'start' ({args['start']})")
        validated_args["title"] = title.strip()
        validated_args["start"] = args["start"]
        validated_args["end"] = args["end"]
        if "attendees" in args:
            att = args["attendees"]
            att_list = [att] if isinstance(att, str) else att
            if not isinstance(att_list, list):
                raise ActionValidationError("Field 'attendees' in calendar.create_event must be a list of emails")
            for em in att_list:
                _validate_email(em, "attendees")
            validated_args["attendees"] = [e.strip() for e in att_list]
        if "location" in args:
            validated_args["location"] = str(args["location"]).strip()
        if "description" in args:
            validated_args["description"] = str(args["description"]).strip()

    elif act_type == "calendar.update_event":
        eid = args["event_id"]
        if not isinstance(eid, str) or not eid.strip() or len(eid) > 100:
            raise ActionValidationError("Field 'event_id' in calendar.update_event must be a string <= 100 chars")
        validated_args["event_id"] = eid.strip()
        # Must have at least one update field
        update_fields = set(args.keys()) - {"event_id"}
        if not update_fields:
            raise ActionValidationError("calendar.update_event requires at least one field to update")
        if "start" in args:
            _validate_iso_datetime(args["start"], "start")
            validated_args["start"] = args["start"]
        if "end" in args:
            _validate_iso_datetime(args["end"], "end")
            validated_args["end"] = args["end"]
        if "start" in validated_args and "end" in validated_args:
            if _validate_iso_datetime(validated_args["end"], "end") <= _validate_iso_datetime(validated_args["start"], "start"):
                raise ActionValidationError("calendar.update_event 'end' must be after 'start'")
        for k in ("title", "location", "description", "status"):
            if k in args:
                validated_args[k] = str(args[k]).strip()
        if "attendees" in args:
            att = args["attendees"]
            att_list = [att] if isinstance(att, str) else att
            if not isinstance(att_list, list):
                raise ActionValidationError("Field 'attendees' in calendar.update_event must be a list of emails")
            for em in att_list:
                _validate_email(em, "attendees")
            validated_args["attendees"] = [e.strip() for e in att_list]

    elif act_type == "reminder.create":
        text = args["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise ActionValidationError("Field 'text' in reminder.create must be a string <= 1000 chars")
        _validate_iso_datetime(args["due"], "due")
        validated_args["text"] = text.strip()
        validated_args["due"] = args["due"]

    elif act_type == "memory.ask":
        q = args["question"]
        if not isinstance(q, str) or not q.strip() or len(q) > 1000:
            raise ActionValidationError("Field 'question' in memory.ask must be a string <= 1000 chars")
        validated_args["question"] = q.strip()

    elif act_type == "app.open":
        app = args["app"]
        if not isinstance(app, str) or not app.strip() or len(app) > 100:
            raise ActionValidationError("Field 'app' in app.open must be a string <= 100 chars")
        validated_args["app"] = app.strip()

    elif act_type == "clarify":
        q = args["question"]
        if not isinstance(q, str) or not q.strip() or len(q) > 1000:
            raise ActionValidationError("Field 'question' in clarify must be a string <= 1000 chars")
        validated_args["question"] = q.strip()

    elif act_type == "confirm":
        s = args["summary"]
        if not isinstance(s, str) or not s.strip() or len(s) > 2000:
            raise ActionValidationError("Field 'summary' in confirm must be a string <= 2000 chars")
        validated_args["summary"] = s.strip()

    return {
        "type": act_type,
        "args": validated_args
    }


def validate_actions_list(actions: Any) -> List[Dict[str, Any]]:
    """Validates a list of actions.
    
    Raises ActionValidationError on any invalid action.
    """
    if not isinstance(actions, list):
        raise ActionValidationError(f"Expected a list of actions, got {type(actions).__name__}")
    if not actions:
        raise ActionValidationError("Empty action list: no valid action produced")
    return [validate_action(act) for act in actions]
