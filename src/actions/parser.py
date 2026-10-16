"""Action parser converting natural language commands into dry-run action calls.

Grounds intents dynamically against Slack users, channels, calendar events, and memory.
Supports Claude CLI (no API key needed, matching eval_harness/llm.py), Anthropic, OpenAI, and Gemini.
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from typing import Dict, List, Any
from datetime import datetime
from src.actions.entities import EntityResolver
from src.retrieval.hybrid import HybridRetriever
from src.ingestion.loader import get_visible_units


def _load_env_if_needed():
    """Load .env file into os.environ if present."""
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip("\"'")


SYSTEM_ACTION_PROMPT = """You are Chronexa's Action Assistant (TextOS).
You translate natural language commands from Alex Rivera into structured dry-run action calls.

Timezone: America/Los_Angeles (-07:00). All datetimes MUST be ISO 8601 with offset, e.g. 2026-09-18T15:00:00-07:00.
Current moment (as_of): {as_of}

ALLOWED ACTION TYPES AND ARGUMENTS:
1. slack.send_message: "to" (Slack user ID like U03SARAHK or channel ID like C10RP), "text"
2. gmail.send: "to" (list of emails), "body" (string), optional "subject", optional "cc" (list)
3. calendar.create_event: "title", "start" (ISO 8601), "end" (ISO 8601), "attendees" (list of emails)
4. calendar.update_event: "event_id", plus fields that change ("start", "end", etc.)
5. reminder.create: "text", "due" (ISO 8601)
6. memory.ask: "question" (used when the command is really asking a question to memory)
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
    def __init__(self, retriever: HybridRetriever = None):
        _load_env_if_needed()
        self.entities = EntityResolver()
        self.retriever = retriever or HybridRetriever()

        try:
            import certifi
            if not os.environ.get("SSL_CERT_FILE"):
                os.environ["SSL_CERT_FILE"] = certifi.where()
        except Exception:
            pass

        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        if self.groq_key:
            self.provider = "groq"
            self.model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
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
        """Isolated Claude Code CLI execution matching eval_harness/llm.py."""
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

    def _call_openai(self, sys_prompt: str, user_prompt: str, retries: int = 5) -> str:
        url = f"{self.base_url}/chat/completions"
        auth_key = self.groq_key if self.provider == "groq" else self.openai_key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_key}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait_time = 3 * (attempt + 1)
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

    def parse(self, command: str, as_of: str) -> List[Dict[str, Any]]:
        as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        visible_units = get_visible_units(as_of_dt)

        # Retrieve relevant memory snippets for context grounding (fast <20ms search)
        mem_hits = self.retriever.search(command, visible_units, top_k=5)
        mem_context = "\n".join([f"- {u.text}" for u, _ in mem_hits])

        known_entities_str = self.entities.format_entities_for_prompt()
        sys_prompt = SYSTEM_ACTION_PROMPT.format(as_of=as_of, known_entities=known_entities_str)
        user_prompt = f"Command: {command}\n\nRelevant Memory Context:\n{mem_context}\n\nJSON Action Response:"

        try:
            if self.provider in ("openai", "groq"):
                raw = self._call_openai(sys_prompt, user_prompt)
            elif self.provider == "anthropic":
                raw = self._call_anthropic(sys_prompt, user_prompt)
            elif self.provider == "gemini":
                raw = self._call_gemini(sys_prompt, user_prompt)
            elif self.provider == "claude-cli":
                raw = self._call_claude_cli(sys_prompt, user_prompt)
            else:
                raw = "{}"

            parsed = json.loads(raw)
            actions = parsed.get("actions", [])
            for act in actions:
                if "args" not in act or not isinstance(act["args"], dict):
                    act["args"] = {}
                for k in list(act.keys()):
                    if k not in ("type", "args"):
                        act["args"][k] = act.pop(k)
            return actions
        except Exception as e:
            print(f"Error parsing action: {e}")
            return []
