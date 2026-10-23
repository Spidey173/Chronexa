"""Hybrid retriever fusing structured lexical BM25 and dense semantic search via Reciprocal Rank Fusion (RRF).

Combines:
- Passage-structured Okapi BM25
- Dense semantic embeddings (BAAI/bge-small-en-v1.5)
- True Reciprocal Rank Fusion with rank parameter k=60
- Query decomposition for multi-faceted questions
- 2-hop iterative retrieval for calendar-anchor and entity-chain queries
- Adjacent-segment injection for meeting context completeness
- Parent-record diversification to guarantee multi-document coverage
- Cross-encoder reranking for deep semantic relevance
- Temporal date boosting via dateparser resolution
"""
import re
from collections import defaultdict
from typing import Any, List, Tuple, Optional
from eval_harness.records import Unit
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        reranker: Optional[Any] = None,
        enable_dense: bool = True,
        enable_rerank: bool = True,
        model_name: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        use_synonyms: bool = True,
        use_legacy_regex: bool = False
    ):
        self.enable_dense = enable_dense
        self.enable_rerank = enable_rerank
        self.dense = dense_retriever
        self.reranker = reranker
        self.use_synonyms = use_synonyms
        self.use_legacy_regex = use_legacy_regex
        if enable_dense and self.dense is None:
            self.dense = DenseRetriever(model_name=model_name)
        if enable_rerank and self.reranker is None:
            try:
                from src.retrieval.reranker import CrossEncoderReranker
                self.reranker = CrossEncoderReranker(model_name=reranker_model)
            except Exception as e:
                self.reranker = None

    def _rrf_fuse(
        self,
        query: str,
        visible_units: List[Unit],
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        bm25_top: int = 80,
        dense_top: int = 80,
    ) -> defaultdict:
        """Core RRF fusion returning {unit_id: rrf_score} dict."""
        bm25_idx = BM25Index(visible_units, use_synonyms=self.use_synonyms)
        bm25_hits = bm25_idx.search(
            query,
            top_k=min(bm25_top, len(visible_units)),
            max_per_record=10,
            use_legacy_regex=self.use_legacy_regex
        )

        rrf_scores = defaultdict(float)
        for rank, (u, _) in enumerate(bm25_hits, start=1):
            rrf_scores[u.id] += bm25_weight / (rrf_k + rank)

        if self.enable_dense and self.dense is not None and getattr(self.dense, "is_available", lambda: True)() and dense_weight > 0.0:
            dense_hits = self.dense.search(query, visible_units, top_k=min(dense_top, len(visible_units)))
            for rank, (u, _) in enumerate(dense_hits, start=1):
                rrf_scores[u.id] += dense_weight / (rrf_k + rank)

        return rrf_scores

    def _resolve_parent(self, u: Unit, unit_by_id: dict) -> Unit:
        """If unit is a Slack reply/reaction citing a context parent, resolve to the parent message."""
        if "Context: [" in u.text and u.id.startswith("SL-F-"):
            try:
                num = int(u.id.split("SL-F-")[1])
                prev_id = f"SL-F-{num - 1:04d}"
                if prev_id in unit_by_id:
                    return unit_by_id[prev_id]
            except ValueError:
                pass
        return u

    def _inject_adjacent_segments(
        self,
        selected: List[Tuple[Unit, float]],
        unit_by_id: dict,
        visible_units: List[Unit],
        max_inject: int = 12,
    ) -> List[Tuple[Unit, float]]:
        """Injects adjacent segments for meeting completeness and parent/reply linkage for Slack messages."""
        selected_ids = {u.id for u, _ in selected}
        injected = []
        injected_per_rec = {}

        # 1. Slack thread and context parents (high precision)
        for u, score in selected:
            if u.id.startswith("SL-F-"):
                try:
                    num = int(u.id.split("SL-F-")[1])
                    prev_id = f"SL-F-{num - 1:04d}"
                    if prev_id not in selected_ids and prev_id in unit_by_id:
                        injected.append((unit_by_id[prev_id], score * 0.98))
                        selected_ids.add(prev_id)
                except ValueError:
                    pass
            for line in u.text.split("\n"):
                if line.startswith("Context: ["):
                    parts = line[len("Context: ["):].split("]:", 1)
                    if len(parts) == 2:
                        spk, body = parts[0].strip(), parts[1].strip()
                        for other in visible_units:
                            if other.id not in selected_ids and spk.lower() in other.text.lower() and body[:35].lower() in other.text.lower():
                                injected.append((other, score * 0.98))
                                selected_ids.add(other.id)
                                break

        # 2. Meeting adjacent segments (limit 2 per meeting)
        for u, score in selected:
            if "#" in u.id and u.id.rsplit("#", 1)[0].startswith("MTG-"):
                prefix, seg_str = u.id.rsplit("#", 1)
                rec_injected = injected_per_rec.get(prefix, 0)
                if rec_injected < 2:
                    try:
                        seg_num = int(seg_str)
                        for delta in [-1, 1]:
                            neighbor_id = f"{prefix}#{seg_num + delta:04d}"
                            if neighbor_id not in selected_ids and neighbor_id in unit_by_id:
                                injected.append((unit_by_id[neighbor_id], score * 0.9))
                                selected_ids.add(neighbor_id)
                                injected_per_rec[prefix] = rec_injected + 1
                                if len(injected) >= max_inject:
                                    return injected + selected
                    except ValueError:
                        pass

        return injected + selected

    def _heuristic_calendar_hop(
        self,
        query: str,
        visible_units: List[Unit],
        hop1_selected: List[Tuple[Unit, float]],
        unit_by_id: dict,
    ) -> List[Tuple[Unit, float]]:
        """Heuristic 2-hop for calendar/travel queries.

        When the query asks about calendar events on a specific day (e.g. 'the day I fly'),
        extract dates from hop-1 evidence and inject calendar events from those dates.
        """
        q_lower = query.lower()
        is_calendar_query = any(w in q_lower for w in [
            "calendar", "schedule", "agenda", "what's on my", "commitments",
            "meetings", "events today", "itinerary"
        ])
        is_travel_query = any(w in q_lower for w in [
            "fly", "flight", "travel", "trip", "denver", "chicago", "new york",
            "board meeting", "offsite"
        ])

        if not (is_calendar_query or is_travel_query):
            return []

        # Extract dates from hop-1 evidence
        try:
            import dateparser
        except ImportError:
            dateparser = None
        from datetime import datetime, timedelta

        dates_found = set()
        for u, _ in hop1_selected[:10]:
            # 1. ISO dates (2026-09-23)
            for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", u.text):
                dates_found.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

            # 2. Full natural dates: "September 23, 2026" or "Sep 23, 2026"
            for m in re.finditer(
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
                r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
                r"\s+(\d{1,2}),?\s+(\d{4})\b",
                u.text, re.I
            ):
                try:
                    if dateparser is not None:
                        dt = dateparser.parse(m.group(0))
                        if dt:
                            dates_found.add(dt.strftime("%Y-%m-%d"))
                    else:
                        clean_str = re.sub(r",", "", m.group(0))
                        dt = datetime.strptime(clean_str, "%B %d %Y" if len(clean_str.split()[0]) > 3 else "%b %d %Y")
                        dates_found.add(dt.strftime("%Y-%m-%d"))
                except Exception:
                    pass

            # 3. Shorter dates: "September 23" without year
            for m in re.finditer(
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
                r"\s+(\d{1,2})\b",
                u.text, re.I
            ):
                try:
                    if dateparser is not None:
                        dt = dateparser.parse(m.group(0), settings={"PREFER_DATES_FROM": "future"})
                        if dt:
                            dates_found.add(dt.strftime("%Y-%m-%d"))
                except Exception:
                    pass

            # 4. Calendar structured time ranges
            for m in re.finditer(r"Time:\s*(\d{4}-\d{2}-\d{2})", u.text):
                dates_found.add(m.group(1))

            # 5. Day-of-week with context date: "Wednesday, September 23"
            for m in re.finditer(
                r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
                r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
                r"\s+\d{1,2}(?:,?\s+\d{4})?)",
                u.text, re.I
            ):
                try:
                    if dateparser is not None:
                        dt = dateparser.parse(m.group(1))
                        if dt:
                            dates_found.add(dt.strftime("%Y-%m-%d"))
                except Exception:
                    pass

        # For travel queries: if we found multi-day event dates (e.g. offsite Sep 24-26),
        # also add the day before (likely travel day)
        if is_travel_query:
            extra_dates = set()
            for d in list(dates_found):
                try:
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    prev_day = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
                    extra_dates.add(prev_day)
                except Exception:
                    pass
            dates_found.update(extra_dates)

        if not dates_found:
            return []


        # Find calendar events and other records on those dates
        injected = []
        existing_ids = {u.id for u, _ in hop1_selected}
        for u in visible_units:
            if u.id in existing_ids:
                continue
            u_date = u.time.strftime("%Y-%m-%d") if u.time else ""
            is_calendar = u.id.startswith("CAL-") or "calendar" in u.text.lower()[:80]
            text_has_date = any(d in u.text for d in dates_found)
            date_match = u_date in dates_found

            if is_calendar and (date_match or text_has_date):
                injected.append((u, 0.025))
                existing_ids.add(u.id)
            elif date_match and text_has_date:
                injected.append((u, 0.015))
                existing_ids.add(u.id)

        return injected

    def search(
        self,
        query: str,
        visible_units: List[Unit],
        top_k: int = 20,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        max_per_record: int = 4,
        as_of: str = "",
    ) -> List[Tuple[Unit, float]]:
        """Searches strictly visible units using Reciprocal Rank Fusion of BM25 and Dense embeddings.

        Guarantees that only units in visible_units can be returned (zero temporal leakage).
        Integrates sub-query decomposition, 2-hop retrieval, adjacent-segment injection,
        multi-stream cross-encoder reranking, and parent record diversification.
        """
        if not visible_units:
            return []

        unit_by_id = {u.id: u for u in visible_units}
        streams = []

        # 1. Primary RRF fusion on the original query
        rrf_scores = self._rrf_fuse(
            query, visible_units, rrf_k=rrf_k,
            bm25_weight=bm25_weight, dense_weight=dense_weight
        )

        # Temporal date boosting
        try:
            from src.retrieval.temporal import resolve_dates_with_dateparser
            from datetime import datetime
            if as_of:
                as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            else:
                as_of_dt = datetime.now()
            resolved_dates = resolve_dates_with_dateparser(query, as_of_dt)
            if resolved_dates:
                for uid in list(rrf_scores.keys()):
                    u = unit_by_id.get(uid)
                    if u:
                        u_date = u.time.strftime("%Y-%m-%d") if u.time else ""
                        if u_date in resolved_dates or any(d in u.text for d in resolved_dates):
                            rrf_scores[uid] *= 1.15
        except Exception:
            pass

        # Primary pool selection
        ranked_uids = sorted(rrf_scores.keys(), key=lambda uid: rrf_scores[uid], reverse=True)
        selected_main = []
        rec_counts_main = defaultdict(int)
        deferred_main = []
        for uid in ranked_uids:
            u = unit_by_id[uid]
            rec_id = u.record if hasattr(u, "record") and u.record else u.id.split("#")[0]
            if rec_counts_main[rec_id] < max_per_record:
                rec_counts_main[rec_id] += 1
                selected_main.append((u, rrf_scores[uid]))
            else:
                deferred_main.append((u, rrf_scores[uid]))
        if len(selected_main) < top_k * 2:
            selected_main.extend(deferred_main[:(top_k * 2 - len(selected_main))])

        selected_main = self._inject_adjacent_segments(selected_main, unit_by_id, visible_units, max_inject=12)
        if self.enable_rerank and self.reranker is not None and len(selected_main) > 0:
            reranked_main = self.reranker.rerank(query, selected_main[:min(45, len(selected_main))], top_k=20)
        else:
            reranked_main = selected_main[:20]
        streams.append([(self._resolve_parent(u, unit_by_id), sc) for u, sc in reranked_main])

        # 2. Sub-query decomposition stream
        try:
            from src.retrieval.decomposition import decompose_query
            decomp = decompose_query(query, as_of or "", use_llm=True)
            sub_queries = decomp.get("sub_queries", [])
            valid_sqs = [sq.strip() for sq in sub_queries if sq.strip().lower() != query.strip().lower() and len(sq.strip()) > 8 and "calculate" not in sq.lower()]
            for sq in valid_sqs[:2]:
                sq_scores = self._rrf_fuse(sq, visible_units, rrf_k=rrf_k, bm25_weight=bm25_weight, dense_weight=dense_weight)
                sq_ranked = sorted(sq_scores.items(), key=lambda x: x[1], reverse=True)
                pool_sq = [(unit_by_id[uid], s) for uid, s in sq_ranked[:30] if uid in unit_by_id]
                pool_sq = self._inject_adjacent_segments(pool_sq, unit_by_id, visible_units, max_inject=10)
                if self.enable_rerank and self.reranker is not None and len(pool_sq) > 0:
                    reranked_sq = self.reranker.rerank(sq, pool_sq[:25], top_k=15)
                else:
                    reranked_sq = pool_sq[:15]
                streams.append([(self._resolve_parent(u, unit_by_id), sc) for u, sc in reranked_sq])
        except Exception:
            pass

        # 3. 2-hop retrieval stream
        try:
            from src.retrieval.decomposition import propose_hop2_query
            from src.retrieval.bm25 import BM25Index
            seen_evidence_recs = set()
            hop1_evidence = []
            for u, sc in reranked_main:
                rec_id = u.record if hasattr(u, "record") and u.record else u.id.split("#")[0]
                if rec_id not in seen_evidence_recs:
                    seen_evidence_recs.add(rec_id)
                    hop1_evidence.append((u, sc))
                if len(hop1_evidence) >= 4:
                    break
            bm25_idx = BM25Index(visible_units)
            for u, sc in bm25_idx.search(query, top_k=6):
                rec_id = u.record if hasattr(u, "record") and u.record else u.id.split("#")[0]
                if rec_id not in seen_evidence_recs:
                    seen_evidence_recs.add(rec_id)
                    hop1_evidence.append((u, sc))
                if len(hop1_evidence) >= 6:
                    break

            hop2_result = propose_hop2_query(query, as_of or "", hop1_evidence, use_llm=True)
            if hop2_result.get("needs_hop2") and hop2_result.get("follow_up_query"):
                follow_up = hop2_result["follow_up_query"].strip()
                hop2_scores = self._rrf_fuse(follow_up, visible_units, rrf_k=rrf_k, bm25_weight=bm25_weight, dense_weight=dense_weight)
                hop2_ranked = sorted(hop2_scores.items(), key=lambda x: x[1], reverse=True)
                pool_h2 = [(unit_by_id[uid], s) for uid, s in hop2_ranked[:30] if uid in unit_by_id]
                pool_h2 = self._inject_adjacent_segments(pool_h2, unit_by_id, visible_units, max_inject=10)
                if self.enable_rerank and self.reranker is not None and len(pool_h2) > 0:
                    reranked_h2 = self.reranker.rerank(follow_up, pool_h2[:25], top_k=15)
                else:
                    reranked_h2 = pool_h2[:15]
                streams.append([(self._resolve_parent(u, unit_by_id), sc) for u, sc in reranked_h2])
        except Exception:
            pass

        # 4. Calendar hop stream
        calendar_injected = self._heuristic_calendar_hop(query, visible_units, reranked_main, unit_by_id)
        if calendar_injected:
            streams.append(calendar_injected)

        # 5. Round-Robin Interleaving with Parent Record Diversification
        final = []
        final_ids = set()
        rec_counts_final = defaultdict(int)
        stream_idx = [0] * len(streams)

        while len(final) < top_k:
            added = False
            for s_i, stream in enumerate(streams):
                idx = stream_idx[s_i]
                while idx < len(stream):
                    u, score = stream[idx]
                    idx += 1
                    rec_id = u.record if hasattr(u, "record") and u.record else u.id.split("#")[0]
                    if u.id not in final_ids and rec_counts_final[rec_id] < max_per_record:
                        final.append((u, score))
                        final_ids.add(u.id)
                        rec_counts_final[rec_id] += 1
                        stream_idx[s_i] = idx
                        added = True
                        break
                stream_idx[s_i] = idx
                if len(final) >= top_k:
                    break
            if not added:
                # If constrained by max_per_record but still need to fill top_k, take remaining
                for s_i, stream in enumerate(streams):
                    idx = stream_idx[s_i]
                    while idx < len(stream) and len(final) < top_k:
                        u, score = stream[idx]
                        idx += 1
                        if u.id not in final_ids:
                            final.append((u, score))
                            final_ids.add(u.id)
                break

        return final[:top_k]
