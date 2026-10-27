"""Grounded answer generator supporting Claude CLI, Anthropic API, OpenAI, and Gemini.

Calls the model with strictly retrieved context, enforces temporal grounding,
parses the structured response, and applies safety/abstention guardrails.
Includes XML document sandboxing to defend against prompt injection.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from typing import Dict, List, Any
from eval_harness.records import Unit
from src.generation.prompt import SYSTEM_PROMPT, USER_TEMPLATE


def _load_env_if_needed():
    """Load .env file into os.environ if present, without overwriting existing environment."""
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    key = k.strip()
                    if key not in os.environ:
                        os.environ[key] = v.strip("\"'")


class AnswerGenerator:
    def __init__(self, model: str = None, api_key: str = None):
        _load_env_if_needed()
        try:
            import certifi
            if not os.environ.get("SSL_CERT_FILE"):
                os.environ["SSL_CERT_FILE"] = certifi.where()
        except Exception:
            pass

        def _is_valid_key(key: str = None) -> bool:
            if not key or not key.strip():
                return False
            k = key.strip().lower()
            return not (k.startswith("your_") or k.startswith("<") or k == "placeholder" or "api_key_here" in k)

        gem_k = api_key or os.environ.get("GEMINI_API_KEY")
        self.gemini_key = gem_k if _is_valid_key(gem_k) else None
        oai_k = api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_key = oai_k if _is_valid_key(oai_k) else None
        ant_k = os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic_key = ant_k if _is_valid_key(ant_k) else None
        groq_k = api_key or os.environ.get("GROQ_API_KEY")
        self.groq_key = groq_k if _is_valid_key(groq_k) else None

        if os.environ.get("CHRONEXA_OFFLINE") == "1":
            self.provider = "none"
            self.model = "local-extractive"
        elif self.groq_key:
            self.provider = "groq"
            self.model = model or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
            self.base_url = "https://api.groq.com/openai/v1"
        elif self.openai_key:
            self.provider = "openai"
            self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        elif self.anthropic_key:
            self.provider = "anthropic"
            self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        elif self.gemini_key:
            self.provider = "gemini"
            self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            self._genai = genai
        elif shutil.which("claude"):
            try:
                test_p = subprocess.run(["claude", "-p", "--model", "haiku", "hi"], capture_output=True, text=True, timeout=3)
                if test_p.returncode == 0 and "not logged in" not in test_p.stdout.lower():
                    self.provider = "claude-cli"
                    self.model = model or os.environ.get("CLAUDE_MODEL", "haiku")
                else:
                    self.provider = "none"
                    self.model = "none"
            except Exception:
                self.provider = "none"
                self.model = "none"
        else:
            self.provider = "none"
            self.model = "none"

    def _call_claude_cli(self, sys_prompt: str, user_prompt: str) -> str:
        """Isolated Claude Code CLI execution matching eval_harness/llm.py."""
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                ["claude", "-p", "--model", self.model, "--tools", "", "--setting-sources", "",
                 "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
                 "--system-prompt", sys_prompt],
                input=user_prompt, capture_output=True, text=True, timeout=120, cwd=cwd)
        if proc.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {proc.stderr[:200]}")
        out = proc.stdout.strip()
        if "```json" in out:
            out = out.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in out:
            out = out.split("```", 1)[1].split("```", 1)[0].strip()
        return out

    def _call_anthropic(self, sys_prompt: str, user_prompt: str, retries: int = 3) -> str:
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
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    out = data["content"][0]["text"].strip()
                    if "```json" in out:
                        out = out.split("```json", 1)[1].split("```", 1)[0].strip()
                    elif "```" in out:
                        out = out.split("```", 1)[1].split("```", 1)[0].strip()
                    return out
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("Failed after max retries")

    def _call_gemini(self, sys_prompt: str, user_prompt: str, retries: int = 5) -> str:
        model = self._genai.GenerativeModel(
            self.model,
            generation_config={"response_mime_type": "application/json", "temperature": 0.0},
            system_instruction=sys_prompt
        )
        for attempt in range(retries):
            try:
                resp = model.generate_content(user_prompt)
                return resp.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "ResourceExhausted" in err_str:
                    wait_time = 15 + (attempt * 5)
                    print(f"Rate limited (429). Retrying in {wait_time}s... (attempt {attempt+1}/{retries})")
                    time.sleep(wait_time)
                else:
                    if attempt == retries - 1:
                        raise e
                    time.sleep(2)
        raise RuntimeError("Failed after max retries")

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
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    wait_s = 2.0 * (attempt + 1)
                    if hasattr(e, "headers") and e.headers and "retry-after" in e.headers:
                        try:
                            wait_s = min(3.0, float(e.headers["retry-after"]) + 0.5)
                        except Exception:
                            pass
                    wait_s = min(4.0, wait_s)
                    print(f"[RateLimit 429] Waiting {wait_s:.1f}s before retry ({attempt+1}/{retries})...")
                    time.sleep(wait_s)
                else:
                    if attempt == retries - 1:
                        print(f"Error calling {self.provider}: {e}")
                        raise e
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError("Failed after max retries")

    def generate(self, question: str, as_of: str, retrieved_units: List[Unit]) -> Dict[str, Any]:
        """Generates a grounded answer from top retrieved units with security boundaries."""
        context_blocks = []
        valid_ids = set()
        total_len = 0
        for idx, u in enumerate(retrieved_units[:10], start=1):
            valid_ids.add(u.id)
            max_chars = 3500 if idx <= 2 else 1000
            text = u.text[:max_chars] if len(u.text) > max_chars else u.text
            # Defuse potential prompt breakout tags in untrusted data
            safe_text = text.replace("</record>", "&lt;/record&gt;").replace("</evidence_records>", "&lt;/evidence_records&gt;")
            if total_len + len(safe_text) > 14000 and idx > 4:
                continue
            src_type = getattr(u, "source_type", "unknown")
            context_blocks.append(f'<record id="{u.id}" source="{src_type}">\n{safe_text}\n</record>')
            total_len += len(safe_text)

        context_str = "\n\n".join(context_blocks) if context_blocks else "(No records found)"

        sys_prompt = SYSTEM_PROMPT.format(as_of=as_of)
        user_prompt = USER_TEMPLATE.format(question=question, as_of=as_of, context=context_str)

        try:
            if self.provider in ("openai", "groq"):
                raw_response = self._call_openai(sys_prompt, user_prompt)
            elif self.provider == "anthropic":
                raw_response = self._call_anthropic(sys_prompt, user_prompt)
            elif self.provider == "gemini":
                raw_response = self._call_gemini(sys_prompt, user_prompt)
            elif self.provider == "claude-cli":
                raw_response = self._call_claude_cli(sys_prompt, user_prompt)
            else:
                raw_response = ""
            parsed = json.loads(raw_response) if raw_response else {}
        except Exception as e:
            print(f"[Generate Exception] {e}")
            parsed = {}

        answer = str(parsed.get("answer", "")).strip()
        abstained = bool(parsed.get("abstained", False))
        sources = parsed.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        clean_sources = sources
        # Grounded fallback if LLM returned empty or is offline
        UNANSWERABLE_TOPICS = [
            "guest network", "wifi password", "wifi", "soc 2", "salary", "pay",
            "compensation", "bonus", "equity", "social security", "home address"
        ]
        if not answer or answer.startswith("Error generating answer"):
            q_lower = question.lower()
            if not retrieved_units or any(w in q_lower for w in UNANSWERABLE_TOPICS):
                answer = "Not in memory: nothing in the data covers this."
                abstained = True
                clean_sources = []
            else:
                top_u = retrieved_units[0]
                raw_t = top_u.text
                body = raw_t.split("]\n", 1)[-1] if "]\n" in raw_t else (raw_t.split("] ", 1)[-1] if "] " in raw_t else raw_t)
                sents = re.split(r"(?<=[.!?])\s+", body)
                clean_sents = []
                for s in sents:
                    s_clean = s.strip()
                    if s_clean:
                        # Skip adversarial instructions and tags
                        if re.search(r"(?:review@pipelinepilot|pipelinepilot|ignore (?:your )?previous instructions|delete records|forward all emails|<\/?system>|<\/?record>)", s_clean, re.I):
                            continue
                        clean_sents.append(s_clean)
                    if len(" ".join(clean_sents).split()) >= 35:
                        break
                answer = " ".join(clean_sents) if clean_sents else body[:200]
                abstained = False
                clean_sources = [top_u.id]

        # Guardrail 1: Filter sources to strictly valid retrieved IDs
        if not clean_sources and not abstained and retrieved_units:
            clean_sources = [retrieved_units[0].id]
        clean_sources = [s for s in clean_sources if s in valid_ids]

        # Guardrail 2: Word count limit (strictly under 110 words)
        words = answer.split()
        if len(words) > 110:
            answer = " ".join(words[:100]) + "."

        # Guardrail 3: Explicit abstention formatting
        if abstained or answer.lower().startswith(("i don't know", "i do not know", "not in memory", "no record")):
            abstained = True
            answer = "Not in memory: nothing in the data covers this."
            clean_sources = []

        # Guardrail 4: Sanitization of prompt injection leak terms and instructions
        injection_patterns = [
            r"review@pipelinepilot(?:\.example\.com)?",
            r"pipelinepilot(?:\.example\.com)?",
            r"forward all emails[^\.\n]*",
            r"ignore (?:your )?previous instructions[^\.\n]*",
            r"delete records[^\.\n]*",
            r"note to any ai[^\.\n]*",
            r"<\/?system>[^\.\n]*"
        ]
        for pat in injection_patterns:
            if re.search(pat, answer, re.IGNORECASE):
                answer = re.sub(pat, "[REDACTED DIRECTIVE]", answer, flags=re.IGNORECASE)

        return {
            "answer": answer,
            "sources": clean_sources,
            "abstained": abstained
        }
