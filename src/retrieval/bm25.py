"""Okapi BM25 implementation with passage structure awareness,
ISO timestamp normalization, temporal metadata indexing, named entity boosting,
universal workplace synonym expansion, and parent record diversification.
"""
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Set, Optional
from eval_harness.records import Unit

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", 
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", 
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", 
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
    "while", "who", "whom", "why", "will", "with", "would"
}

MONTH_MAP = {
    "01": ["january", "jan"], "02": ["february", "feb"], "03": ["march", "mar"],
    "04": ["april", "apr"], "05": ["may"], "06": ["june", "jun"],
    "07": ["july", "jul"], "08": ["august", "aug"], "09": ["september", "sep", "sept"],
    "10": ["october", "oct"], "11": ["november", "nov"], "12": ["december", "dec"]
}

MONTH_REV = {
    "jan": "01", "january": "01", "feb": "02", "february": "02", "mar": "03", "march": "03",
    "apr": "04", "april": "04", "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10", "nov": "11", "november": "11", "dec": "12", "december": "12"
}

WORKPLACE_SYNONYMS = {
    "contract": ["contract", "agreement", "order", "terms", "proposal"],
    "agreement": ["contract", "agreement", "order", "terms"],
    "signed": ["signed", "executed", "agreement", "contract", "review", "decision"],
    "sign": ["sign", "close", "agree", "contract", "deal"],
    "slip": ["slip", "delay", "delayed", "push", "pushed", "postpone", "regression", "move", "bug", "blocker", "issue", "fix", "broke"],
    "delay": ["delay", "delayed", "slip", "slipped", "pushed", "blocker", "move", "regression", "bug", "issue", "fix"],
    "slipped": ["slip", "delay", "regression", "move", "bug", "blocker"],
    "flight": ["flight", "fly", "airline", "travel", "ticket", "airport", "trip"],
    "fly": ["flight", "fly", "airline", "travel", "ticket", "trip"],
    "travel": ["travel", "trip", "flight", "fly", "airline", "visit", "transit"],
    "email": ["email", "gmail", "thread", "message", "sent"],
    "mail": ["email", "gmail", "message", "sent"],
    "sent": ["sent", "email", "delivered", "send", "out"],
    "call": ["call", "meeting", "sync", "huddle", "meet", "calendar"],
    "meeting": ["meeting", "call", "sync", "huddle", "standup", "calendar", "board"],
    "calendar": ["calendar", "event", "invite", "schedule", "agenda", "board"],
    "launch": ["launch", "release", "ship", "deploy", "rollout", "date"],
    "ship": ["launch", "release", "ship", "deploy"],
    "pricing": ["pricing", "price", "rate", "cost", "quote", "tier", "proposal"],
    "proposal": ["proposal", "pricing", "quote", "contract", "terms"],
    "dictate": ["dictate", "dictation", "memo", "note"],
    "dictation": ["dictate", "dictation", "memo", "note"],
    "hire": ["hire", "hiring", "req", "headcount", "recruiting", "role"],
    "hiring": ["hire", "hiring", "req", "headcount", "recruiting", "role"],
    "latency": ["latency", "p95", "p50", "median", "performance", "milliseconds", "seconds", "response"],
    "p95": ["p95", "latency", "performance", "percentile"],
    "routing": ["routing", "route", "planner", "navigation", "directions"],
    "board": ["board", "meeting", "quarterly", "director", "executive", "calendar"],
}

try:
    from py_rust_stemmers import SnowballStemmer
    _stemmer = SnowballStemmer("english")
    def stem_word(w: str) -> str:
        return _stemmer.stem_word(w)
except ImportError:
    def stem_word(w: str) -> str:
        for sfx in ("ing", "tion", "ed", "es", "s"):
            if len(w) > len(sfx) + 3 and w.endswith(sfx):
                return w[:-len(sfx)]
        return w


def normalize_text(text: str) -> str:
    """Normalize ISO 8601 timestamps so e.g. 2026-09-23T09:00:00 separates into 2026-09-23 09:00:00."""
    return re.sub(r"(\d{4}-\d{2}-\d{2})[tT](\d{2})", r"\1 \2", text)


def extract_dates(text: str) -> List[str]:
    """Extracts and expands dates from ISO (YYYY-MM-DD) and English expressions (Sep 10)."""
    expanded = []
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        year, month, day = m.group(1), m.group(2), m.group(3)
        expanded.extend([year, month, day, str(int(day))])
        if month in MONTH_MAP:
            expanded.extend(MONTH_MAP[month])
    for m in re.finditer(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})\b", text, re.I):
        m_name, day = m.group(1).lower(), m.group(2)
        expanded.extend([m_name, day, f"{int(day):02d}"])
        if m_name in MONTH_REV:
            expanded.append(MONTH_REV[m_name])
    return expanded


def extract_entities(query: str) -> Set[str]:
    """Identifies capitalized proper nouns / entities (excluding query-initial words)."""
    words = re.findall(r"\b[A-Za-z0-9_-]+\b", query)
    entities = set()
    for idx, w in enumerate(words):
        if idx > 0 and w[0].isupper() and w.lower() not in STOPWORDS:
            entities.add(stem_word(w.lower()))
    return entities


_PASSAGE_TOKEN_CACHE: Dict[Tuple[str, int], Tuple[List[str], List[str]]] = {}


def get_passage_tokens(unit: Unit) -> Tuple[List[str], List[str]]:
    """Cached passage tokenization to prevent redundant stemmer/regex loops."""
    key = (unit.id, hash(unit.text))
    if key not in _PASSAGE_TOKEN_CACHE:
        _PASSAGE_TOKEN_CACHE[key] = tokenize_passage(unit)
    return _PASSAGE_TOKEN_CACHE[key]


def tokenize_passage(unit: Unit) -> Tuple[List[str], List[str]]:
    """Splits unit text into header metadata tokens and body content tokens."""
    text = normalize_text(unit.text)
    m = re.match(r"^(\[[^\]]+\]\s*)+", text)
    if m:
        header_text = m.group(0)
        body_text = text[m.end():]
    else:
        header_text = ""
        body_text = text

    h_tokens = [stem_word(t) for t in re.findall(r"[a-zA-Z0-9]+", header_text.lower()) if t not in STOPWORDS]
    h_tokens.extend([stem_word(t) for t in extract_dates(header_text)])

    b_tokens = [stem_word(t) for t in re.findall(r"[a-zA-Z0-9]+", body_text.lower()) if t not in STOPWORDS]
    b_tokens.extend([stem_word(t) for t in extract_dates(body_text)])

    # Index unit.time metadata directly into body tokens
    if hasattr(unit, "time") and unit.time:
        iso_date = unit.time.strftime("%Y-%m-%d")
        b_tokens.extend([stem_word(t) for t in extract_dates(iso_date)])

    return h_tokens, b_tokens


def tokenize_query(query: str, use_synonyms: bool = True) -> List[str]:
    """Tokenizes query with stemming, date expansion, and universal workplace synonyms."""
    norm_q = normalize_text(query)
    raw = re.findall(r"[a-zA-Z0-9]+", norm_q.lower())
    tokens = []
    seen_stems = set()
    for t in raw:
        if t not in STOPWORDS:
            s = stem_word(t)
            if s not in seen_stems:
                tokens.append(s)
                seen_stems.add(s)
            if use_synonyms and t in WORKPLACE_SYNONYMS:
                for syn in WORKPLACE_SYNONYMS[t]:
                    syn_s = stem_word(syn)
                    if syn_s not in seen_stems:
                        tokens.append(syn_s)
                        seen_stems.add(syn_s)
    for d in extract_dates(query):
        s = stem_word(d)
        if s not in seen_stems:
            tokens.append(s)
            seen_stems.add(s)
    return tokens


def decompose_query(query: str) -> List[str]:
    """Decomposes compound or multi-hop natural language queries into distinct sub-queries."""
    sub_queries = [query]
    
    # 1. Temporal delta: "How many days after X did I Y"
    m = re.search(r"days after (?:the )?(.*?)(?:did I|did we|I|we)\s+(.*)", query, re.I)
    if m:
        sub_queries.append(m.group(1).strip())
        sub_queries.append(m.group(2).strip())
        
    # 2. Sent confirmation: "What did I dictate to X on Y, and did it go out"
    m = re.search(r"(.*?)\s+and\s+(?:did it go out|was it sent)", query, re.I)
    if m:
        sub_queries.append(m.group(1).strip())
        sub_queries.append("sent email confirmation out")
        
    # 3. Slip cause: "Why did X slip / delay from [date]"
    m = re.search(r"(?:why did|what caused)\s+(.*?)\s+(?:slip|delay|move|push)\s+(?:from\s+.*)?", query, re.I)
    if m:
        sub_queries.append(f"{m.group(1)} slip delay move regression issue")
        
    # 4. Calendar event anchor: "What's on my calendar the day I X"
    m = re.search(r"(?:what's on my calendar|calendar).*?(?:the day I|when I|day of)\s+(.*)", query, re.I)
    if m:
        sub_queries.append(m.group(1).strip())
        
    return list(dict.fromkeys(sub_queries))


class BM25Index:
    def __init__(self, units: List[Unit], k1: float = 1.5, b: float = 0.4, use_synonyms: bool = True):
        self.units = units
        self.k1 = k1
        self.b = b
        self.N = len(units)
        self.use_synonyms = use_synonyms
        
        self.doc_lens = []
        self.doc_freqs = Counter()
        self.doc_body_counts = []
        self.doc_head_counts = []
        
        total_len = 0
        for u in units:
            h_toks, b_toks = get_passage_tokens(u)
            n_tokens = len(h_toks) + len(b_toks)
            self.doc_lens.append(n_tokens)
            total_len += n_tokens
            
            b_cnt = Counter(b_toks)
            h_cnt = Counter(h_toks)
            self.doc_body_counts.append(b_cnt)
            self.doc_head_counts.append(h_cnt)
            
            seen = set(b_cnt.keys()) | set(h_cnt.keys())
            for t in seen:
                self.doc_freqs[t] += 1
                
        self.avgdl = (total_len / self.N) if self.N > 0 else 1.0
        self.idf = {}
        for t, freq in self.doc_freqs.items():
            self.idf[t] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

    def search_tokens(self, q_tokens: List[str], entity_tokens: Set[str] = None) -> List[Tuple[int, float]]:
        if not q_tokens:
            return []
        scores = [0.0] * self.N
        body_matches = [0] * self.N
        entity_matches = [0] * self.N
        
        for t in q_tokens:
            if t not in self.idf:
                continue
            idf_val = self.idf[t]
            is_entity = bool(entity_tokens and t in entity_tokens)
            term_weight = 1.8 if is_entity else 1.0
            
            for i in range(self.N):
                tf_b = self.doc_body_counts[i].get(t, 0)
                tf_h = self.doc_head_counts[i].get(t, 0)
                if tf_b > 0:
                    body_matches[i] += 1
                if is_entity and (tf_b > 0 or tf_h > 0):
                    entity_matches[i] += 1
                
                # Weight body matches 1.0, header context matches 0.35
                effective_tf = tf_b * 1.0 + tf_h * 0.35
                if effective_tf > 0:
                    doc_len = self.doc_lens[i]
                    denom = effective_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[i] += term_weight * idf_val * (effective_tf * (self.k1 + 1.0)) / denom

        for i in range(self.N):
            # Heavily discount header-only matches if query contains content tokens
            if scores[i] > 0 and body_matches[i] == 0:
                scores[i] *= 0.35
            # If query had named entities, penalize candidates that match none of them
            if entity_tokens and entity_matches[i] == 0:
                scores[i] *= 0.5

        ranked = [(i, scores[i]) for i in range(self.N) if scores[i] > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def search(
        self,
        query: str,
        top_k: int = 20,
        max_per_record: int = 2,
        use_legacy_regex: bool = False
    ) -> List[Tuple[Unit, float]]:
        """Multi-facet search supporting both clean token-based search and legacy v2 regex mode."""
        if not use_legacy_regex:
            toks = tokenize_query(query, use_synonyms=self.use_synonyms)
            entities = extract_entities(query)
            hits = self.search_tokens(toks, entities)
            selected = []
            record_counts = defaultdict(int)
            deferred = []
            for i, score in hits:
                u = self.units[i]
                rec_id = u.record if hasattr(u, "record") and u.record else u.id.split("#")[0]
                if record_counts[rec_id] < max_per_record:
                    record_counts[rec_id] += 1
                    selected.append((u, score))
                else:
                    deferred.append((u, score))
            if len(selected) < top_k:
                selected.extend(deferred[:(top_k - len(selected))])
            return selected[:top_k]

        sub_queries = decompose_query(query)
        entities = extract_entities(query)
        rrf = defaultdict(float)
        RRF_K = 40
        
        for sq in sub_queries:
            toks = tokenize_query(sq, use_synonyms=self.use_synonyms)
            sq_entities = extract_entities(sq) or entities
            hits = self.search_tokens(toks, sq_entities)
            w = 1.0 if sq == query else 0.85
            rec_counts = defaultdict(int)
            rank = 1
            for idx, _ in hits:
                u = self.units[idx]
                rec_id = u.record if hasattr(u, "record") and u.record else u.id.split("#")[0]
                if rec_counts[rec_id] < 2:
                    rec_counts[rec_id] += 1
                    rrf[idx] += w / (RRF_K + rank)
                    rank += 1

        # Multi-hop anchor resolution for calendar / event anchor queries
        if re.search(r"calendar.*?(?:day I|when I|day of)", query, re.I):
            top_ranked = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:3]
            for tidx in top_ranked:
                u = self.units[tidx]
                dates = extract_dates(u.text)
                if dates:
                    cal_toks = [stem_word("calendar")] + [stem_word(d) for d in dates]
                    cal_hits = self.search_tokens(cal_toks)
                    for rank, (cidx, _) in enumerate(cal_hits[:10], start=1):
                        cu = self.units[cidx]
                        if "calendar" in cu.text.lower() or cu.id.startswith("CAL-"):
                            rrf[cidx] += 1.5 / (RRF_K + rank)

        ranked_indices = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)
        
        # Passage diversification by parent record
        selected = []
        record_counts = defaultdict(int)
        deferred = []
        
        for i in ranked_indices:
            u = self.units[i]
            rec_id = u.record if hasattr(u, "record") and u.record else u.id.split("#")[0]
            if record_counts[rec_id] < max_per_record:
                record_counts[rec_id] += 1
                selected.append((u, rrf[i]))
            else:
                deferred.append((u, rrf[i]))
                
        if len(selected) < top_k:
            selected.extend(deferred[:(top_k - len(selected))])
            
        return selected[:top_k]
