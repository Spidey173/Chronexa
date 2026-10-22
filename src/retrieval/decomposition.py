"""Generic LLM query decomposition and 2-hop iterative retrieval.

- Decomposes compound/multi-faceted queries into 1-3 sub-queries, extracted entities, and temporal refs.
- Supports generic two-hop retrieval: inspects hop-1 evidence to propose follow-up search or date filters.
- Persists outputs on disk in .cache/decomposition so evaluation runs offline without breaking run.sh.
- Includes a robust heuristic fallback if LLM client is unavailable or returns an error.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import dateparser
from eval_harness.records import Unit
from src.retrieval.temporal import resolve_dates_with_dateparser, DATE_INDICATORS

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache" / "decomposition"
QUERY_CACHE_FILE = CACHE_DIR / "query_decomposition.json"
TWOHOP_CACHE_FILE = CACHE_DIR / "two_hop_decomposition.json"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_cache(cache_file: Path) -> Dict[str, Any]:
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache_file: Path, cache: Dict[str, Any]):
    _ensure_cache_dir()
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to write cache {cache_file}: {e}")


def _call_repo_llm(prompt: str, system: str = "") -> Optional[str]:
    """Attempts to call available LLM client from the repository."""
    if os.environ.get("CHRONEXA_OFFLINE") == "1":
        return None
    # 1. Try AnswerGenerator providers (OpenAI, Anthropic, Gemini, or authenticated Claude CLI)
    try:
        from src.generation.generator import AnswerGenerator
        gen = AnswerGenerator()
        if gen.provider in ("openai", "groq"):
            sys_prompt = system or "You are a search query decomposition engine. Output strictly valid JSON."
            return gen._call_openai(sys_prompt, prompt)
        elif gen.provider == "anthropic":
            sys_prompt = system or "You are a search query decomposition engine. Output strictly valid JSON."
            return gen._call_anthropic(sys_prompt, prompt)
        elif gen.provider == "gemini":
            sys_prompt = system or "You are a search query decomposition engine. Output strictly valid JSON."
            return gen._call_gemini(sys_prompt, prompt)
        elif gen.provider == "claude-cli":
            from eval_harness.llm import ask
            out = ask(prompt, system=system or "You are a concise JSON-only assistant.")
            if out and not out.startswith("ERROR:") and "not logged in" not in out.lower():
                return out
    except Exception:
        pass

    return None


def _fallback_decompose(query: str, as_of: str) -> Dict[str, Any]:
    """Robust, generic rule-free fallback query analyzer."""
    sub_queries = [query]
    
    # 1. Generic compound clause splitting on conjunctions and question boundaries
    # e.g., "what did I dictate to X AND did it go out?"
    parts = re.split(r"\s+(?:and|also|while|but)\s+|\s*;\s*", query, flags=re.I)
    if len(parts) > 1:
        for p in parts:
            p_clean = p.strip()
            if len(p_clean) > 8 and p_clean.lower() not in [sq.lower() for sq in sub_queries]:
                sub_queries.append(p_clean)

    # 2. Extract capitalized entity words
    words = re.findall(r"\b[A-Za-z0-9_-]+\b", query)
    entities = []
    for idx, w in enumerate(words):
        if idx > 0 and w[0].isupper() and w.lower() not in {"what", "who", "when", "where", "why", "did", "was", "is", "are"}:
            if w not in entities:
                entities.append(w)

    # 3. Extract temporal indicators
    time_refs = [w for w in words if w.lower() in DATE_INDICATORS]

    return {
        "sub_queries": sub_queries[:3],
        "entities": entities,
        "time_references": time_refs
    }


def decompose_query(query: str, as_of: str, use_llm: bool = True) -> Dict[str, Any]:
    """Decomposes question into 1-3 sub-queries, extracted entities, and temporal refs.

    Checks disk cache first. If not cached, attempts LLM client. Falls back safely if offline.
    """
    cache_key = f"{query.strip()}||{as_of.strip()}"
    cache = _load_cache(QUERY_CACHE_FILE)
    if cache_key in cache:
        return cache[cache_key]

    result = None
    if use_llm:
        sys_msg = "You are an enterprise search query decomposition engine. You output only valid JSON."
        user_msg = (
            f"Question: \"{query}\"\n"
            f"Reference time (as_of): \"{as_of}\"\n\n"
            "Task: Break down this question into search components.\n"
            "1. sub_queries: list of 1 to 3 distinct search queries covering each aspect needed.\n"
            "2. entities: list of named entities (people, organizations, projects, apps).\n"
            "3. time_references: list of date/time expressions mentioned.\n\n"
            "Output JSON strictly with keys \"sub_queries\", \"entities\", \"time_references\"."
        )
        raw = _call_repo_llm(user_msg, system=sys_msg)
        if raw:
            try:
                # Clean JSON code fences if present
                clean = raw
                if "```json" in clean:
                    clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in clean:
                    clean = clean.split("```", 1)[1].split("```", 1)[0].strip()
                data = json.loads(clean)
                if isinstance(data.get("sub_queries"), list) and data["sub_queries"]:
                    result = {
                        "sub_queries": [str(sq) for sq in data["sub_queries"]][:3],
                        "entities": [str(e) for e in data.get("entities", [])],
                        "time_references": [str(t) for t in data.get("time_references", [])]
                    }
            except Exception:
                result = None

    if not result:
        result = _fallback_decompose(query, as_of)

    # Save to disk cache
    cache[cache_key] = result
    _save_cache(QUERY_CACHE_FILE, cache)
    return result


def _fallback_two_hop(query: str, as_of: str, hop1_hits: List[Tuple[Unit, float]]) -> Dict[str, Any]:
    """Generic heuristic fallback for 2-hop retrieval."""
    q_lower = query.lower()
    needs_calendar = any(w in q_lower for w in ["calendar", "schedule", "agenda", "commitments", "meetings", "itinerary", "events"])
    
    # Check if hop 1 revealed an anchor date in passage body content
    date_candidates = []
    for u, _ in hop1_hits[:5]:
        body = u.text.split("]\n", 1)[-1] if "]\n" in u.text else u.text.split("] ", 1)[-1]
        for m in re.finditer(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b", body, re.I):
            try:
                dt = dateparser.parse(m.group(0))
                if dt:
                    date_candidates.append(dt.strftime("%Y-%m-%d"))
            except Exception:
                pass
        for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", body):
            date_candidates.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    if needs_calendar and date_candidates:
        target_date = date_candidates[0]
        return {
            "needs_hop2": True,
            "follow_up_query": f"calendar schedule events meetings {target_date}",
            "target_date": target_date,
            "target_entity": None
        }

    return {
        "needs_hop2": False,
        "follow_up_query": None,
        "target_date": None,
        "target_entity": None
    }


def propose_hop2_query(
    query: str,
    as_of: str,
    hop1_hits: List[Tuple[Unit, float]],
    use_llm: bool = True
) -> Dict[str, Any]:
    """Inspects Hop 1 evidence to propose a generic Hop 2 follow-up query or date filter."""
    if not hop1_hits:
        return {"needs_hop2": False, "follow_up_query": None, "target_date": None, "target_entity": None}

    cache_key = f"{query.strip()}||{as_of.strip()}"
    cache = _load_cache(TWOHOP_CACHE_FILE)
    if cache_key in cache:
        return cache[cache_key]

    result = None
    if use_llm:
        snippets = "\n".join([f"[{idx+1}] (ID: {u.id}) {u.text[:500]}" for idx, (u, _) in enumerate(hop1_hits[:6])])
        sys_msg = "You are a multi-hop enterprise search assistant. You output only valid JSON."
        user_msg = (
            f"Original Question: \"{query}\"\n"
            f"Reference time (as_of): \"{as_of}\"\n\n"
            f"Top Evidence Retrieved from Hop 1:\n{snippets}\n\n"
            "Task: Determine if answering the original question requires a follow-up retrieval hop to find primary sources, incident reports, bug tickets, team chat messages, or connected calendar/flight records.\n"
            "IMPORTANT: In enterprise search, even if the evidence above mentions a reason (e.g. in a narrative, chat draft, or meeting), we ALWAYS need primary sources (original Slack conversations, tickets, emails) for verification.\n"
            "If the evidence mentions specific causes, entities, technical terms, or bugs (e.g. 'geocoding regression', a person's name, a ticket ID, or an address issue), you MUST set needs_hop2 to true and your follow_up_query MUST use those specific terms from the evidence (e.g. 'geocoding regression Canada Mexico addresses Route Planner v2').\n\n"
            "Respond strictly in JSON format:\n"
            "{\n"
            "  \"needs_hop2\": true or false,\n"
            "  \"follow_up_query\": \"specific entity/cause-based query from evidence\" (or null),\n"
            "  \"target_date\": \"YYYY-MM-DD\" (or null),\n"
            "  \"target_entity\": \"entity name\" (or null)\n"
            "}"
        )
        raw = _call_repo_llm(user_msg, system=sys_msg)
        if raw:
            try:
                clean = raw
                if "```json" in clean:
                    clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in clean:
                    clean = clean.split("```", 1)[1].split("```", 1)[0].strip()
                data = json.loads(clean)
                if "needs_hop2" in data:
                    result = {
                        "needs_hop2": bool(data.get("needs_hop2", False)),
                        "follow_up_query": str(data["follow_up_query"]) if data.get("follow_up_query") else None,
                        "target_date": str(data["target_date"]) if data.get("target_date") else None,
                        "target_entity": str(data["target_entity"]) if data.get("target_entity") else None
                    }
            except Exception:
                result = None

    if not result:
        result = _fallback_two_hop(query, as_of, hop1_hits)

    cache[cache_key] = result
    _save_cache(TWOHOP_CACHE_FILE, cache)
    return result
