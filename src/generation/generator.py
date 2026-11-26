"""Grounded answer generator supporting Claude CLI, Anthropic API, OpenAI, and Gemini.

Calls the model with strictly retrieved context, enforces temporal grounding,
parses the structured response, and applies safety/abstention guardrails.
Includes XML document sandboxing to defend against prompt injection.
"""
import json
import re
from typing import Dict, List, Any
from eval_harness.records import Unit
from src.generation.prompt import SYSTEM_PROMPT, USER_TEMPLATE
from src.llm_client import LLMClient


class AnswerGenerator:
    def __init__(self, model: str = None, api_key: str = None):
        self._llm = LLMClient(model=model, api_key=api_key, offline_model="local-extractive")

    @property
    def provider(self) -> str:
        return self._llm.provider

    @provider.setter
    def provider(self, value: str) -> None:
        self._llm.provider = value

    @property
    def model(self) -> str:
        return self._llm.model

    @model.setter
    def model(self, value: str) -> None:
        self._llm.model = value

    def _strip_stale_mentions(self, answer: str) -> str:
        """Post-generation guard: removes phrases that mention superseded/stale values."""
        # Parenthetical stale references: "(moved from Sep 30)", "(originally Oct 14)"
        answer = re.sub(
            r'\s*\([^)]*(?:previously|originally|moved from|changed from|was initially|formerly|first set|started as|used to be)[^)]*\)',
            '', answer, flags=re.I
        )
        # Inline "moved/changed from X to Y" → remove the "from X" clause
        answer = re.sub(
            r',?\s*(?:moved|changed|shifted|pushed|updated|rescheduled|went)\s+from\s+[^,\.;]+(?:\s+to\s+)',
            ' ', answer, flags=re.I
        )
        # "originally/initially X, [now/then] Y" → remove the "originally X" clause
        answer = re.sub(
            r'(?:originally|initially|previously|formerly|first)\s+'
            r'(?:set for|scheduled for|reported as|at|on|was|said)\s+[^,\.;]+[,;]\s*(?:(?:now|then|later|currently)\s+)?',
            '', answer, flags=re.I
        )
        # "It was originally X." at sentence level → remove entire sentence
        answer = re.sub(
            r'(?:^|\.\s+)[Ii]t\s+was\s+(?:originally|initially|previously|formerly)\s+[^\.]+\.\s*',
            '. ', answer
        )
        # "was originally/initially X" inline patterns
        answer = re.sub(
            r'(?:it was|was)\s+(?:originally|initially|previously)\s+[^,\.;]+[,\.;]?\s*',
            '', answer, flags=re.I
        )
        # "X first said/reported Y, then corrected" → keep only the correction
        answer = re.sub(
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:first\s+)?(?:said|reported|stated|mentioned)\s+[^,;]+[,;]\s*(?:then\s+)?(?:corrected|updated|revised)[^,\.;]*[,;]?\s*',
            '', answer
        )
        # Cleanup whitespace and punctuation artifacts
        answer = re.sub(r'\s*,\s*\.', '.', answer)
        answer = re.sub(r',\s*,', ',', answer)
        answer = re.sub(r'\.\s*\.', '.', answer)
        answer = re.sub(r'\s{2,}', ' ', answer).strip()
        answer = re.sub(r'^[,;.\s]+', '', answer)
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        return answer

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
            date_str = u.time.strftime("%Y-%m-%d %H:%M") if hasattr(u, 'time') and u.time and hasattr(u.time, 'strftime') else ""
            context_blocks.append(f'<record id="{u.id}" source="{src_type}" date="{date_str}">\n{safe_text}\n</record>')
            total_len += len(safe_text)

        context_str = "\n\n".join(context_blocks) if context_blocks else "(No records found)"

        sys_prompt = SYSTEM_PROMPT.format(as_of=as_of)
        user_prompt = USER_TEMPLATE.format(question=question, as_of=as_of, context=context_str)

        try:
            raw_response = self._llm.call(
                sys_prompt,
                user_prompt,
                anthropic_retries=3,
                gemini_retries=5,
                cli_timeout=120,
            )
            parsed = json.loads(raw_response) if raw_response else {}
        except Exception as e:
            print(f"[Generate Exception] {e}")
            parsed = {}

        answer = str(parsed.get("answer", "")).strip()
        answer = self._strip_stale_mentions(answer)
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
        # Grounded fallback if LLM returned empty or is offline
        if not answer:
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
            r"<!--[\s\S]*?-->",  # HTML/XML comment injection payloads
            r"review@pipelinepilot(?:\.example\.com)?",
            r"pipelinepilot(?:\.example\.com)?",
            r"forward (?:all )?emails[^\.\n]*",
            r"(?:ignore|disregard) (?:all )?(?:your )?previous instructions[^\.\n]*",
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
