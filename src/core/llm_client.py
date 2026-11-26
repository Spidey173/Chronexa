"""Shared LLM provider detection, environment loading, and HTTP/CLI client."""
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Optional


def load_env_if_needed() -> None:
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


def is_valid_api_key(key: Optional[str]) -> bool:
    """Return True when key looks like a real credential rather than a template placeholder."""
    if not key or not key.strip():
        return False
    k = key.strip().lower()
    return not (
        k.startswith("your_")
        or k.startswith("<")
        or k == "placeholder"
        or "api_key_here" in k
    )


def strip_json_fences(text: str) -> str:
    """Strip surrounding ```json or ``` markdown code fences if present."""
    out = text.strip()
    if "```json" in out:
        return out.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in out:
        return out.split("```", 1)[1].split("```", 1)[0].strip()
    return out


def ensure_certifi_ssl() -> None:
    """Ensure SSL_CERT_FILE points to certifi bundle if not already configured."""
    try:
        import certifi

        if not os.environ.get("SSL_CERT_FILE"):
            os.environ["SSL_CERT_FILE"] = certifi.where()
    except Exception:
        pass


class LLMClient:
    """Unified multi-provider LLM client (Groq, OpenAI, Anthropic, Gemini, Claude CLI)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_gemini_first: bool = False,
        offline_model: str = "none",
    ):
        load_env_if_needed()
        ensure_certifi_ssl()

        gem_k = api_key or os.environ.get("GEMINI_API_KEY")
        self.gemini_key = gem_k if is_valid_api_key(gem_k) else None
        oai_k = api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_key = oai_k if is_valid_api_key(oai_k) else None
        ant_k = os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic_key = ant_k if is_valid_api_key(ant_k) else None
        groq_k = api_key or os.environ.get("GROQ_API_KEY")
        self.groq_key = groq_k if is_valid_api_key(groq_k) else None

        self.base_url: Optional[str] = None
        self._genai = None

        if os.environ.get("CHRONEXA_OFFLINE") == "1":
            self.provider = "none"
            self.model = offline_model
        elif self.groq_key:
            self.provider = "groq"
            self.model = model or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
            self.base_url = "https://api.groq.com/openai/v1"
        elif self.openai_key:
            self.provider = "openai"
            self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self.base_url = os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/")
        elif self.anthropic_key:
            self.provider = "anthropic"
            self.model = model or os.environ.get(
                "ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"
            )
        elif self.gemini_key:
            self.provider = "gemini"
            self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_key)
            self._genai = genai
        elif shutil.which("claude"):
            try:
                test_p = subprocess.run(
                    ["claude", "-p", "--model", "haiku", "hi"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if (
                    test_p.returncode == 0
                    and "not logged in" not in test_p.stdout.lower()
                ):
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

    def call_claude_cli(
        self, sys_prompt: str, user_prompt: str, timeout: int = 60
    ) -> str:
        """Execute prompt via isolated Claude Code CLI."""
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                [
                    "claude",
                    "-p",
                    "--model",
                    self.model,
                    "--tools",
                    "",
                    "--setting-sources",
                    "",
                    "--strict-mcp-config",
                    "--disable-slash-commands",
                    "--no-session-persistence",
                    "--system-prompt",
                    sys_prompt,
                ],
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {proc.stderr[:200]}")
        return strip_json_fences(proc.stdout)

    def call_anthropic(
        self, sys_prompt: str, user_prompt: str, retries: int = 1
    ) -> str:
        """Call Anthropic Messages API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": self.model,
            "max_tokens": 1024,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.0,
        }
        data_bytes = json.dumps(body).encode("utf-8")
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url, data=data_bytes, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return strip_json_fences(data["content"][0]["text"])
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("Failed after max retries")

    def call_openai(
        self, sys_prompt: str, user_prompt: str, retries: int = 2
    ) -> str:
        """Call OpenAI-compatible chat completions API (OpenAI or Groq)."""
        url = f"{self.base_url}/chat/completions"
        auth_key = self.groq_key if self.provider == "groq" else self.openai_key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_key}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chronexa/2.0",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        data_bytes = json.dumps(body).encode("utf-8")
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url, data=data_bytes, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or getattr(e, "code", None) == 429:
                    wait_s = 2.0 * (attempt + 1)
                    if hasattr(e, "headers") and e.headers and "retry-after" in e.headers:
                        try:
                            wait_s = min(3.0, float(e.headers["retry-after"]) + 0.5)
                        except Exception:
                            pass
                    wait_s = min(4.0, wait_s)
                    time.sleep(wait_s)
                else:
                    if attempt == retries - 1:
                        raise e
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError("Failed after max retries")

    def call_gemini(
        self, sys_prompt: str, user_prompt: str, retries: int = 1
    ) -> str:
        """Call Google Gemini GenerativeModel API."""
        model = self._genai.GenerativeModel(
            self.model,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
            system_instruction=sys_prompt,
        )
        for attempt in range(retries):
            try:
                resp = model.generate_content(user_prompt)
                return resp.text
            except Exception as e:
                err_str = str(e)
                if (
                    "429" in err_str
                    or "quota" in err_str.lower()
                    or "ResourceExhausted" in err_str
                ):
                    if attempt == retries - 1:
                        raise e
                    time.sleep(15 + (attempt * 5))
                else:
                    if attempt == retries - 1:
                        raise e
                    time.sleep(2)
        raise RuntimeError("Failed after max retries")

    def call(
        self,
        sys_prompt: str,
        user_prompt: str,
        anthropic_retries: int = 1,
        gemini_retries: int = 1,
        cli_timeout: int = 60,
    ) -> str:
        """Dispatch prompt to the active LLM provider."""
        if self.provider in ("openai", "groq"):
            return self.call_openai(sys_prompt, user_prompt)
        if self.provider == "anthropic":
            return self.call_anthropic(sys_prompt, user_prompt, retries=anthropic_retries)
        if self.provider == "gemini":
            return self.call_gemini(sys_prompt, user_prompt, retries=gemini_retries)
        if self.provider == "claude-cli":
            return self.call_claude_cli(sys_prompt, user_prompt, timeout=cli_timeout)
        return ""
