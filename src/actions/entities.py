"""Entity resolution component for Chronexa Action Assistant.

Maps natural language names, channels, and event references to their
canonical IDs from data/connectors (Slack, Calendar, Gmail).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from src.config import DATA_DIR


class EntityResolver:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.users: List[Dict] = []
        self.channels: List[Dict] = []
        self.events: List[Dict] = []
        self.contacts: Dict[str, str] = {}  # name / handle -> email
        self._load()

    def _load(self):
        # 1. Slack Users
        users_file = self.data_dir / "connectors/slack/users.json"
        if users_file.exists():
            with open(users_file, "r", encoding="utf-8") as f:
                self.users = json.load(f)
                for u in self.users:
                    email = u.get("email")
                    if email:
                        real = u.get("real_name", "").lower().strip()
                        handle = u.get("name", "").lower().strip()
                        if real:
                            self.contacts[real] = email
                            first_name = real.split()[0]
                            if first_name not in self.contacts:
                                self.contacts[first_name] = email
                        if handle:
                            self.contacts[handle] = email

        # 2. Slack Channels
        chans_file = self.data_dir / "connectors/slack/channels.json"
        if chans_file.exists():
            with open(chans_file, "r", encoding="utf-8") as f:
                self.channels = json.load(f)

        # 3. Google Calendar Events
        events_file = self.data_dir / "connectors/google_calendar/events.jsonl"
        if events_file.exists():
            with open(events_file, "r", encoding="utf-8") as f:
                self.events = [json.loads(line) for line in f if line.strip()]

        # 4. Gmail Messages for discovering external contacts
        gmail_file = self.data_dir / "connectors/gmail/messages.jsonl"
        if gmail_file.exists():
            with open(gmail_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    m = json.loads(line)
                    # Extract from 'from', 'to', 'cc'
                    for addr_str in [m.get("from")] + m.get("to", []) + m.get("cc", []):
                        if not addr_str:
                            continue
                        if "<" in addr_str and ">" in addr_str:
                            name_part = addr_str.split("<")[0].strip("\"' ")
                            email_part = addr_str.split("<")[1].split(">")[0].strip()
                            if name_part and email_part:
                                nl = name_part.lower()
                                self.contacts[nl] = email_part
                                first = nl.split()[0]
                                if first not in self.contacts:
                                    self.contacts[first] = email_part
                                self.contacts[email_part.lower()] = email_part

    def resolve_slack_user(self, name: str) -> Optional[str]:
        """Resolves a person's name or handle to their Slack User ID."""
        q = name.lower().strip()
        for u in self.users:
            real = u.get("real_name", "").lower().strip()
            handle = u.get("name", "").lower().strip()
            if q == real or q == handle or q == real.split()[0]:
                return u["id"]
        return None

    def resolve_slack_channel(self, name: str) -> Optional[str]:
        """Resolves a channel name to its channel ID."""
        clean = name.lstrip("#").lower().strip()
        for c in self.channels:
            if clean == c.get("name", "").lower().strip():
                return c["id"]
        return None

    def resolve_email(self, name: str) -> Optional[str]:
        """Resolves any internal or external contact to their primary email."""
        q = name.lower().strip()
        if q in self.contacts:
            return self.contacts[q]
        # Partial match
        for k, v in self.contacts.items():
            if q in k or k in q:
                return v
        return None

    def find_event(self, query: str) -> Optional[Dict]:
        """Finds the most relevant calendar event by token overlap."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        best_ev = None
        max_matches = 0
        for ev in self.events:
            summary = ev.get("summary", "").lower()
            s_tokens = set(re.findall(r"\w+", summary))
            matches = len(q_tokens & s_tokens)
            if matches > max_matches:
                max_matches = matches
                best_ev = ev
        return best_ev

    def format_entities_for_prompt(self) -> str:
        """Dynamically builds the KNOWN ENTITIES block from actual indexed data."""
        lines = ["KNOWN ENTITIES:"]
        
        # Slack Users
        lines.append("- Slack Users:")
        for u in self.users:
            real = u.get("real_name", "")
            handle = u.get("name", "")
            uid = u.get("id", "")
            email = u.get("email", "")
            title = u.get("profile", {}).get("title", "")
            title_str = f" ({title})" if title else ""
            lines.append(f"  - {real}{title_str}: {uid} ({email})")
            
        # Slack Channels
        lines.append("- Slack Channels:")
        for c in self.channels:
            name = c.get("name", "")
            cid = c.get("id", "")
            purpose = c.get("purpose", {}).get("value", "")
            purp_str = f" ({purpose})" if purpose else ""
            lines.append(f"  - #{name}: {cid}{purp_str}")
            
        # External Contacts
        slack_emails = {u.get("email", "").lower() for u in self.users if u.get("email")}
        lines.append("- External Contacts:")
        seen_ext = set()
        for name, email in sorted(self.contacts.items()):
            if "@" in email and email.lower() not in slack_emails and "@" not in name:
                if email not in seen_ext:
                    seen_ext.add(email)
                    lines.append(f"  - {name.title()}: {email} (contact via Email, NOT in Slack)")
                    
        # Calendar Events
        lines.append("- Known Calendar Events:")
        for ev in self.events:
            summary = ev.get("summary", "")
            eid = ev.get("id", "")
            st = ev.get("start", {}).get("dateTime", "")
            en = ev.get("end", {}).get("dateTime", "")
            status = ev.get("status", "")
            status_str = f", status: {status}" if status and status != "confirmed" else ""
            lines.append(f"  - {summary}: event_id {eid} ({st} to {en}{status_str})")
            
        return "\n".join(lines)
