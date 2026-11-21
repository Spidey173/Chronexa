"""Action executor executing proposals via the configured integration adapter."""
from typing import Dict, List, Any, Optional
from src.actions.adapters import get_adapter, IntegrationAdapter


class ActionExecutionError(Exception):
    pass


class ActionExecutor:
    def __init__(self, adapter: Optional[IntegrationAdapter] = None):
        self.adapter = adapter or get_adapter()

    def execute_single(self, action: Dict[str, Any], user_credentials: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        act_type = action.get("type", "")
        args = action.get("args", {})

        if act_type == "slack.send_message":
            return self.adapter.send_slack_message(
                to=args.get("to", ""),
                text=args.get("text", ""),
                credentials=user_credentials
            )
        elif act_type == "gmail.send":
            return self.adapter.send_gmail(
                to=args.get("to", []),
                body=args.get("body", ""),
                subject=args.get("subject"),
                cc=args.get("cc"),
                bcc=args.get("bcc"),
                credentials=user_credentials
            )
        elif act_type == "calendar.create_event":
            return self.adapter.create_calendar_event(
                title=args.get("title", ""),
                start=args.get("start", ""),
                end=args.get("end", ""),
                attendees=args.get("attendees"),
                description=args.get("description"),
                credentials=user_credentials
            )
        elif act_type == "calendar.update_event":
            return self.adapter.update_calendar_event(
                event_id=args.get("event_id", ""),
                start=args.get("start"),
                end=args.get("end"),
                title=args.get("title"),
                credentials=user_credentials
            )
        elif act_type == "reminder.create":
            return self.adapter.create_reminder(
                text=args.get("text", ""),
                due=args.get("due", ""),
                credentials=user_credentials
            )
        elif act_type == "app.open":
            return self.adapter.open_app(
                app=args.get("app", ""),
                credentials=user_credentials
            )
        elif act_type in ("clarify", "confirm", "memory.ask"):
            # Informational dry-run actions
            return {"status": "skipped", "type": act_type, "note": "Informational action requires user input"}
        else:
            raise ActionExecutionError(f"Unsupported action type for execution: {act_type}")

    def execute(self, payload: Any, user_credentials: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Executes a list of actions or single action dictionary."""
        actions = payload if isinstance(payload, list) else [payload]
        results = []
        for act in actions:
            res = self.execute_single(act, user_credentials=user_credentials)
            results.append(res)
        return results
