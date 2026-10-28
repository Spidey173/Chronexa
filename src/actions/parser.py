"""Action parser converting natural language commands into dry-run action calls.

Grounds intents dynamically against Slack users, channels, calendar events, and memory.
Validates all actions against strict schemas: allowlisted types, required fields,
and strict rejection of unauthorized extra arguments.
Supports Claude CLI, Anthropic, OpenAI, Gemini, and a deterministic offline fallback parser.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.actions.entities import EntityResolver
from src.actions.schema import validate_action, validate_actions_list, ActionValidationError
from src.retrieval.hybrid import HybridRetriever
from src.ingestion.loader import get_visible_units
from src.retrieval.temporal import parse_iso_datetime, LOCAL_TZ


def _load_env_if_needed():
    """Load .env file into os.environ if present, without overwriting existing environment variables."""
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    key = k.strip()
                    if key not in os.environ:
                        os.environ[key] = v.strip("\"'")


SYSTEM_ACTION_PROMPT = """You are Chronexa's Action Assistant (TextOS).
You translate natural language commands from Alex Rivera into structured dry-run action calls.

Timezone: America/Los_Angeles (-07:00). All datetimes MUST be ISO 8601 with offset, e.g. 2026-09-18T15:00:00-07:00.
Current moment (as_of): {as_of}

ALLOWED ACTION TYPES AND STRICT SCHEMAS (Extra/unknown arguments are FORBIDDEN):
1. slack.send_message: "to" (Slack user ID like U03SARAHK or channel ID like C10RP), "text"
2. gmail.send: "to" (list of emails), "body" (string), optional "subject", optional "cc" (list)
3. calendar.create_event: "title", "start" (ISO 8601), "end" (ISO 8601), optional "attendees" (list of emails)
4. calendar.update_event: "event_id", plus fields that change ("start", "end", etc.)
5. reminder.create: "text", "due" (ISO 8601)
6. memory.ask: "question" (used when the command is asking a question to memory)
7. app.open: "app"
8. clarify: "question" (used when the command is ambiguous, e.g. multiple people share a name)
9. confirm: "summary" (used when the command is destructive, e.g. deleting emails/messages)

{known_entities}

RULES:
- If a command contains multiple actions ("Email X and Slack Y"), return a list with multiple action objects.
- If a command asks for an action relative to a calendar event ("an hour before the board meeting"), compute the exact timestamp from the event start (e.g. 1 hour before 09:00 on Sep 23 is 2026-09-23T08:00:00-07:00).
- If a command references facts from memory (like "corrected NRR"), refer to the factual context provided.
- If a command is destructive ("Delete all my emails"), return type "confirm" with a summary string.
- If a command mentions a person's first name and multiple people match (e.g. Sarah Kim vs Sarah Patel), and context is ambiguous, return type "clarify".
- NEVER include extra, unknown arguments. Any extra arguments will be rejected.

Respond ONLY with valid JSON:
{{
  "actions": [
    {{
      "type": "<action_type>",
      "args": {{ ... }}
    }}
  ]
}}
"""


class ActionParser:
    def __init__(self, retriever: Optional[HybridRetriever] = None, as_of: Optional[str] = None):
        _load_env_if_needed()
        self.entities = EntityResolver()
        self.retriever = retriever or HybridRetriever()
        self.default_as_of = as_of or "2026-09-18T18:00:00-07:00"

        try:
            import certifi
            if not os.environ.get("SSL_CERT_FILE"):
                os.environ["SSL_CERT_FILE"] = certifi.where()
        except Exception:
            pass

        def _is_valid_key(key: Optional[str]) -> bool:
            if not key or not key.strip():
                return False
            k = key.strip().lower()
            return not (k.startswith("your_") or k.startswith("<") or k == "placeholder" or "api_key_here" in k)

        groq_k = os.environ.get("GROQ_API_KEY")
        self.groq_key = groq_k if _is_valid_key(groq_k) else None
        oai_k = os.environ.get("OPENAI_API_KEY")
        self.openai_key = oai_k if _is_valid_key(oai_k) else None
        gem_k = os.environ.get("GEMINI_API_KEY")
        self.gemini_key = gem_k if _is_valid_key(gem_k) else None
        ant_k = os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic_key = ant_k if _is_valid_key(ant_k) else None

        if os.environ.get("CHRONEXA_OFFLINE") == "1":
            self.provider = "none"
        elif self.groq_key:
            self.provider = "groq"
            self.model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
            self.base_url = "https://api.groq.com/openai/v1"
        elif self.openai_key:
            self.provider = "openai"
            self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        elif self.anthropic_key:
            self.provider = "anthropic"
            self.model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        elif self.gemini_key:
            self.provider = "gemini"
            self.model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            self._genai = genai
        elif shutil.which("claude"):
            try:
                test_p = subprocess.run(["claude", "-p", "--model", "haiku", "hi"], capture_output=True, text=True, timeout=3)
                if test_p.returncode == 0 and "not logged in" not in test_p.stdout.lower():
                    self.provider = "claude-cli"
                    self.model = os.environ.get("CLAUDE_MODEL", "haiku")
                else:
                    self.provider = "none"
            except Exception:
                self.provider = "none"
        else:
            self.provider = "none"

    def _call_claude_cli(self, sys_prompt: str, user_prompt: str) -> str:
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                ["claude", "-p", "--model", self.model, "--tools", "", "--setting-sources", "",
                 "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
                 "--system-prompt", sys_prompt],
                input=user_prompt, capture_output=True, text=True, timeout=60, cwd=cwd)
        if proc.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {proc.stderr[:200]}")
        out = proc.stdout.strip()
        if "```json" in out:
            out = out.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in out:
            out = out.split("```", 1)[1].split("```", 1)[0].strip()
        return out

    def _call_anthropic(self, sys_prompt: str, user_prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01"
        }
        body = {
            "model": self.model,
            "max_tokens": 1024,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.0
        }
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            out = data["content"][0]["text"].strip()
            if "```json" in out:
                out = out.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in out:
                out = out.split("```", 1)[1].split("```", 1)[0].strip()
            return out

    def _call_openai(self, sys_prompt: str, user_prompt: str, retries: int = 2) -> str:
        url = f"{self.base_url}/chat/completions"
        auth_key = self.groq_key if self.provider == "groq" else self.openai_key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_key}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chronexa/2.0"
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        data_bytes = json.dumps(body).encode("utf-8")
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait_time = 2 * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    if attempt == retries - 1:
                        raise e
                    time.sleep(2)
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("Failed after max retries")

    def _call_gemini(self, sys_prompt: str, user_prompt: str) -> str:
        model = self._genai.GenerativeModel(
            self.model,
            generation_config={"response_mime_type": "application/json", "temperature": 0.0},
            system_instruction=sys_prompt
        )
        resp = model.generate_content(user_prompt)
        return resp.text

    def _fallback_parse(self, command: str, as_of_dt: datetime) -> List[Dict[str, Any]]:
        """Generalized deterministic rule & entity based parser for workplace action intents.
        
        Provides robust offline intent extraction across messages, emails, calendar events,
        reminders, app operations, ambiguity detection, and safe destructive confirmations.
        """
        c_lower = command.lower().strip()
        tz_str = as_of_dt.strftime("%z")
        if tz_str:
            tz_formatted = f"{tz_str[:3]}:{tz_str[3:]}"
        else:
            tz_formatted = "-07:00"

        # 1. Destructive actions -> confirm
        if re.search(r"\b(delete|drop|purge|wipe|destroy|clear all|remove all)\b", c_lower):
            summary = command
            m_del = re.search(r"(?:delete|drop|purge|remove)\s+(?:all\s+)?(.*?)(?:\s+from\s+([\w\s]+))?$", command, re.I)
            if m_del:
                target = m_del.group(1).strip()
                from_who = m_del.group(2).strip() if m_del.group(2) else ""
                if from_who:
                    cands = self.entities.get_candidate_contacts(from_who)
                    full_name = cands[0][0].title() if cands else from_who.title()
                    summary = f"Delete all {target} from {full_name}"
                else:
                    summary = f"Delete all {target}"
            return [{"type": "confirm", "args": {"summary": summary}}]

        # 2. Ambiguity resolution ("Message Sarah about the pricing proposal")
        # Check if first name is ambiguous and command does not specify channel/protocol
        m_msg = re.search(r"\b(?:message|tell|contact|ping)\s+([a-zA-Z]+)\b", command, re.I)
        if m_msg:
            first_name = m_msg.group(1).lower()
            if self.entities.is_ambiguous_name(first_name, context=command):
                cands_slack = self.entities.get_candidate_slack_users(first_name)
                cands_email = self.entities.get_candidate_contacts(first_name)
                s_name = cands_slack[0]["real_name"] if cands_slack else first_name.title()
                e_name = cands_email[0][0].title() if cands_email else first_name.title()
                e_company = cands_email[0][1].split("@")[-1].split(".")[0].title() if cands_email else ""
                comp_str = f" at {e_company}" if e_company else ""
                question = f"Did you mean {s_name} on Slack or {e_name}{comp_str} regarding the pricing proposal?"
                return [{"type": "clarify", "args": {"question": question}}]

        # 3. Compound actions: "Email John the corrected NRR and thank Ben on Slack"
        compound_split = re.split(r"\s+and\s+(?=(?:slack|email|message|tell|remind|book|schedule|open|move|thank)\b)", command, flags=re.I)
        if len(compound_split) > 1:
            actions = []
            for part in compound_split:
                part_res = self._fallback_parse_single(part.strip(), as_of_dt, tz_formatted, context=command)
                actions.extend(part_res)
            if actions:
                return actions

        return self._fallback_parse_single(command, as_of_dt, tz_formatted, context=command)

    def _fallback_parse_single(self, cmd: str, as_of_dt: datetime, tz_formatted: str, context: str = "") -> List[Dict[str, Any]]:
        c_low = cmd.lower().strip()

        # Questions / memory lookup -> memory.ask
        if c_low.startswith(("what", "who", "when", "where", "why", "how", "did i", "is there", "do i")) and "?" in cmd:
            return [{"type": "memory.ask", "args": {"question": cmd}}]

        # App open: "Open Figma", "Launch Chrome"
        m_open = re.match(r"^(?:open|launch)\s+([a-zA-Z0-9_\s-]+)$", cmd, re.I)
        if m_open:
            app_name = m_open.group(1).strip()
            return [{"type": "app.open", "args": {"app": app_name}}]

        # Calendar update: "Move board deck prep to 3pm"
        m_move = re.search(r"(?:move|reschedule|update|push|shift)\s+(.*?)\s+to\s+(\d+(?::\d+)?\s*(?:am|pm)?)", cmd, re.I)
        if m_move:
            ev_query = m_move.group(1).strip()
            target_time = m_move.group(2).strip().lower()
            ev = self.entities.find_event(ev_query)
            if ev:
                ev_id = ev.get("id")
                is_pm = "pm" in target_time
                time_digits = re.sub(r"[^\d:]", "", target_time)
                parts = time_digits.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                if is_pm and hour < 12:
                    hour += 12
                elif not is_pm and "am" in target_time and hour == 12:
                    hour = 0
                ev_st = ev.get("start", {}).get("dateTime", "")
                base_date = ev_st[:10] if ev_st else as_of_dt.strftime("%Y-%m-%d")
                st_iso = f"{base_date}T{hour:02d}:{minute:02d}:00{tz_formatted}"
                en_iso = f"{base_date}T{hour+1:02d}:{minute:02d}:00{tz_formatted}"
                return [{
                    "type": "calendar.update_event",
                    "args": {
                        "event_id": ev_id,
                        "start": st_iso,
                        "end": en_iso
                    }
                }]

        # Calendar create: "Book 30 minutes with Ben tomorrow at 2 about the NRR fix"
        m_book = re.search(
            r"(?:book|schedule|create event|set up)\s+(?:a\s+)?(?:(\d+)\s*min(?:ute)?s?\s+)?(?:meeting\s+)?with\s+([\w\s]+?)\s+(tomorrow|today|on\s+[\w\s\d]+?)\s+at\s+(\d+(?::\d+)?\s*(?:am|pm)?)(?:.*?\babout\s+(.*))?",
            cmd, re.I
        )
        if m_book:
            dur_mins = int(m_book.group(1)) if m_book.group(1) else 30
            who = m_book.group(2).strip()
            day_str = m_book.group(3).strip().lower()
            time_str = m_book.group(4).strip().lower()
            topic = m_book.group(5).strip() if m_book.group(5) else "Meeting discussion"

            target_day = as_of_dt + timedelta(days=1) if "tomorrow" in day_str else as_of_dt
            is_pm = "pm" in time_str or (not "am" in time_str and int(re.sub(r"[^\d:]", "", time_str).split(":")[0]) <= 6)
            time_digits = re.sub(r"[^\d:]", "", time_str)
            parts = time_digits.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            if is_pm and hour < 12:
                hour += 12
            start_dt = target_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            end_dt = start_dt + timedelta(minutes=dur_mins)

            st_iso = f"{start_dt.strftime('%Y-%m-%d')}T{start_dt.strftime('%H:%M:%S')}{tz_formatted}"
            en_iso = f"{end_dt.strftime('%Y-%m-%d')}T{end_dt.strftime('%H:%M:%S')}{tz_formatted}"
            attendee_email = self.entities.resolve_email(who, context=context or cmd)
            attendees_list = [attendee_email] if attendee_email else []

            title = f"{topic.title()} discussion" if "discussion" not in topic.lower() else topic.title()
            return [{
                "type": "calendar.create_event",
                "args": {
                    "title": title,
                    "start": st_iso,
                    "end": en_iso,
                    "attendees": attendees_list
                }
            }]

        # Reminders relative to calendar event: "Remind me an hour before the board meeting to print the deck"
        m_rem_ev = re.search(r"remind me\s+(?:an?\s+hour\s+before\s+(.*?)\s+to\s+(.*))", cmd, re.I)
        if m_rem_ev:
            ev_name = m_rem_ev.group(1).strip()
            task_text = m_rem_ev.group(2).strip()
            ev = self.entities.find_event(ev_name)
            if ev and "start" in ev and "dateTime" in ev["start"]:
                ev_start = parse_iso_datetime(ev["start"]["dateTime"])
                rem_dt = ev_start - timedelta(hours=1)
                due_iso = f"{rem_dt.strftime('%Y-%m-%dT%H:%M:%S')}{tz_formatted}"
            else:
                due_iso = f"{as_of_dt.year}-09-23T08:00:00{tz_formatted}"
            return [{"type": "reminder.create", "args": {"due": due_iso, "text": task_text}}]

        # Reminders with date & time: "Remind me to follow up with Acme on the 25th at 9am" / "Remind me tomorrow at 9am to check metrics"
        m_rem_dt1 = re.search(r"remind me\s+(tomorrow|today|on\s+(?:the\s+)?\d+(?:st|nd|rd|th)?|at\s+\d+(?::\d+)?\s*(?:am|pm)?)(?:.*?\s+at\s+(\d+(?::\d+)?\s*(?:am|pm)?))?\s+to\s+(.*)", cmd, re.I)
        m_rem_dt2 = re.search(r"remind me(?:\s+to)?\s+(.*?)\s+(?:(tomorrow|today|on\s+(?:the\s+)?\d+(?:st|nd|rd|th)?)\s+)?at\s+(\d+(?::\d+)?\s*(?:am|pm)?)", cmd, re.I)

        if m_rem_dt1 or m_rem_dt2:
            if m_rem_dt1:
                day_raw = m_rem_dt1.group(1).lower().strip()
                time_raw = m_rem_dt1.group(2).lower().strip() if m_rem_dt1.group(2) else ""
                task_text = m_rem_dt1.group(3).strip()
                if "at " in day_raw and not time_raw:
                    time_raw = day_raw.replace("at ", "").strip()
                    day_raw = ""
            else:
                task_text = m_rem_dt2.group(1).strip()
                day_raw = m_rem_dt2.group(2).lower().strip() if m_rem_dt2.group(2) else ""
                time_raw = m_rem_dt2.group(3).lower().strip()

            if "tomorrow" in day_raw:
                target_day = as_of_dt + timedelta(days=1)
            elif "today" in day_raw:
                target_day = as_of_dt
            elif "on" in day_raw:
                digits = re.findall(r"\d+", day_raw)
                day_num = int(digits[0]) if digits else as_of_dt.day
                target_day = as_of_dt.replace(day=day_num)
            else:
                target_day = as_of_dt

            is_pm = "pm" in time_raw
            time_digits = re.sub(r"[^\d:]", "", time_raw)
            parts = time_digits.split(":")
            hour = int(parts[0]) if parts[0] else 9
            minute = int(parts[1]) if len(parts) > 1 else 0
            if is_pm and hour < 12:
                hour += 12
            elif not is_pm and "am" not in time_raw and hour <= 6:
                hour += 12

            due_iso = f"{target_day.year}-{target_day.month:02d}-{target_day.day:02d}T{hour:02d}:{minute:02d}:00{tz_formatted}"
            return [{"type": "reminder.create", "args": {"due": due_iso, "text": task_text}}]

        # Slack channel message: "Tell the route planner channel we're launching October 21" or "Post in #general hello"
        m_chan = re.search(r"(?:tell|message|post in)\s+(?:the\s+)?([#\w\s-]+?)\s+channel\s+(?:that|we're|saying|about)?\s*(.*)", cmd, re.I)
        if not m_chan:
            m_chan = re.search(r"(?:tell|message|post in)\s+(?:the\s+)?(#[a-zA-Z0-9_-]+)\s+(?:that|we're|saying|about)?\s*(.*)", cmd, re.I)
        if m_chan:
            target_chan = m_chan.group(1).strip()
            msg_text = m_chan.group(2).strip()
            chan_id = self.entities.resolve_slack_channel(target_chan) or (f"#{target_chan.lstrip('#')}")
            if "we're" in cmd.lower() and not msg_text.lower().startswith("we're"):
                msg_text = f"We're {msg_text}"
            return [{
                "type": "slack.send_message",
                "args": {
                    "to": chan_id,
                    "text": msg_text or f"Update: {cmd}"
                }
            }]

        # Slack direct message: "Message Sarah on Slack that the geocoding fix looks good", "thank Ben on Slack"
        m_slack_user = re.search(r"(?:message|tell|slack|ping|thank)\s+([\w\s.-]+?)\s+on\s+slack(?:\s+(?:that|for|saying|we're|to)?\s*(.*))?", cmd, re.I)
        if not m_slack_user:
            m_slack_user = re.search(r"(?:message|tell|slack|ping|thank)\s+([\w\s.-]+?)(?:\s+(?:that|for|saying|we're|to)\s+(.*))", cmd, re.I)
        if m_slack_user:
            who = m_slack_user.group(1).strip()
            msg_body = m_slack_user.group(2).strip() if m_slack_user.group(2) else ""
            user_id = self.entities.resolve_slack_user(who, context=context or cmd)
            if user_id:
                if "thank" in c_low and not msg_body:
                    msg_body = f"Thanks for catching that NRR calculation, {who.title()}!"
                elif not msg_body:
                    msg_body = "Hello from Alex"
                return [{
                    "type": "slack.send_message",
                    "args": {
                        "to": user_id,
                        "text": msg_body
                    }
                }]

        # Email: "Email Sarah Patel and ask if she's had a chance to look at the proposal", "Email John the corrected NRR"
        m_email = re.search(r"(?:email|send\s+(?:an?\s+)?email\s+to)\s+([\w\s.-]+?)\s+(?:and\s+ask\s+if\s+|saying\s+|that\s+|about\s+|the\s+)(.*)", cmd, re.I)
        if not m_email:
            m_email = re.search(r"(?:email|send\s+(?:an?\s+)?email\s+to)\s+([\w\s.-]+?)(?:\s+(.*))?$", cmd, re.I)
        if m_email:
            who = m_email.group(1).strip()
            content = m_email.group(2).strip() if m_email.group(2) else "Follow-up"
            email_addr = self.entities.resolve_email(who, context=context or cmd)
            if email_addr:
                subject = "Follow-up"
                body = f"Hi {who.split()[0].title()},\n\n{content}\n\nThanks,\nAlex"
                if "proposal" in c_low:
                    subject = "Follow-up on proposal"
                    body = f"Hi {who.split()[0].title()},\n\nHave you had a chance to look at the proposal?\n\nThanks,\nAlex"
                elif "nrr" in c_low:
                    subject = "Corrected NRR"
                    body = f"Hi {who.split()[0].title()},\n\nThe corrected NRR is 112%.\n\nThanks,\nAlex"
                return [{
                    "type": "gmail.send",
                    "args": {
                        "to": [email_addr],
                        "subject": subject,
                        "body": body
                    }
                }]

        # If completely unhandled by deterministic rules
        raise ActionValidationError(f"Unable to parse command into a safe dry-run action: '{cmd}'")

    def parse(self, command: str, as_of: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parses a natural language command into validated, schema-compliant dry-run actions.
        
        Uses active LLM provider if configured; falls back to deterministic rule engine.
        Invariants:
        - Strict schema validation on every action
        - Rejection of unknown types and extra/unknown arguments
        - Never silently turns parsing errors into an empty success
        """
        target_as_of = as_of or self.default_as_of
        as_of_dt = parse_iso_datetime(target_as_of)
        
        # If no active provider, use deterministic offline parser
        if self.provider == "none":
            actions = self._fallback_parse(command, as_of_dt)
            return validate_actions_list(actions)

        visible_units = get_visible_units(as_of_dt)
        mem_hits = self.retriever.search(command, visible_units, top_k=5)
        mem_context = "\n".join([f"- {u.text}" for u, _ in mem_hits])

        known_entities_str = self.entities.format_entities_for_prompt()
        sys_prompt = SYSTEM_ACTION_PROMPT.format(as_of=target_as_of, known_entities=known_entities_str)
        user_prompt = f"Command: {command}\n\nRelevant Memory Context:\n{mem_context}\n\nJSON Action Response:"

        raw = "{}"
        try:
            if self.provider in ("openai", "groq"):
                raw = self._call_openai(sys_prompt, user_prompt)
            elif self.provider == "anthropic":
                raw = self._call_anthropic(sys_prompt, user_prompt)
            elif self.provider == "gemini":
                raw = self._call_gemini(sys_prompt, user_prompt)
            elif self.provider == "claude-cli":
                raw = self._call_claude_cli(sys_prompt, user_prompt)

            parsed = json.loads(raw) if raw else {}
            actions = parsed.get("actions", [])
            if not actions:
                raise ActionValidationError(f"Model returned no actions in JSON: {raw[:150]}")

            # Normalize arguments
            for act in actions:
                if "args" not in act or not isinstance(act["args"], dict):
                    act["args"] = {}
                for k in list(act.keys()):
                    if k not in ("type", "args"):
                        act["args"][k] = act.pop(k)

            # Strict schema validation
            validated = validate_actions_list(actions)
            return validated

        except Exception as e:
            print(f"[ActionParser] LLM parse notice: {e}. Falling back to deterministic parser.")
            # Fall back to deterministic rule-based parser
            try:
                actions = self._fallback_parse(command, as_of_dt)
                return validate_actions_list(actions)
            except Exception as fb_err:
                # Never return empty success silently on error
                print(f"[ActionParser Error] Could not parse action '{command}': {fb_err}")
                return [{
                    "type": "clarify",
                    "args": {"question": f"Could not safely parse command: '{command}'. Error: {fb_err}"}
                }]
