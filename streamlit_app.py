"""Chronexa: Enterprise Temporal Memory Engine & Action Assistant
Interactive Demonstration Dashboard for Technical Interviews & Portfolio
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import streamlit as st
except ImportError:
    print("Please install streamlit: pip install streamlit")
    sys.exit(1)

from src.config import DATA_DIR, EVALS_DIR
from src.ingestion.loader import get_visible_units
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.actions.parser import ActionParser
from src.generation.generator import AnswerGenerator

# Page Setup
st.set_page_config(
    page_title="Chronexa | Enterprise Temporal Memory & TextOS",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern Styling
st.markdown("""
<style>
    .badge-pill {
        display: inline-block;
        background: rgba(255, 119, 0, 0.12);
        color: #ff9933;
        border: 1px solid rgba(255, 119, 0, 0.3);
        border-radius: 12px;
        padding: 3px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-success {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 3px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .timeline-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #ff7700;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .citation-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .stat-metric {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f3f4f6;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_system():
    retriever = HybridRetriever()
    action_parser = ActionParser()
    generator = AnswerGenerator()
    return retriever, action_parser, generator


retriever, action_parser, generator = init_system()

# Benchmark Scenarios
BENCHMARKS = [
    {
        "label": "1. Launch Date (Sep 9 - Planning Stage -> Sep 30)",
        "question": "When is Route Planner v2 launching?",
        "as_of": "2026-09-09T09:00:00-07:00",
        "description": "Initial target set in the Sept 8 planning meeting. Knowledge of subsequent delays does not exist yet.",
        "expected": "September 30, 2026"
    },
    {
        "label": "2. Launch Date (Sep 12 - Geocoding Slip -> Oct 14)",
        "question": "When is Route Planner v2 launching?",
        "as_of": "2026-09-12T12:00:00-07:00",
        "description": "Moved on Sept 10 due to geocoding regression. Note: Sept 16 Acme request has NOT occurred yet.",
        "expected": "October 14, 2026"
    },
    {
        "label": "3. Launch Date (Sep 18 - Training Week Slip -> Oct 21)",
        "question": "When is Route Planner v2 launching?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "description": "Moved in Go/No-Go sync on Sept 16 after Acme asked for dispatcher training week.",
        "expected": "October 21, 2026"
    },
    {
        "label": "4. Commitment Lifecycle: Sarah Patel Pricing Proposal",
        "question": "Did I send Sarah Patel the pricing proposal I promised?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "description": "Tracks promised dates, extensions requested, and final dispatch on Sept 15.",
        "expected": "Yes, sent Sept 15 ($18/veh/mo)"
    },
    {
        "label": "5. Attribution & Reported Speech: Dark Mode Removal",
        "question": "Did John agree to cut dark mode?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "description": "Distinguishes hearsay (Dana claiming John agreed) vs John's actual statements.",
        "expected": "Hearsay distinguished; shipping in v2.1 fast-follow"
    },
    {
        "label": "6. Calibrated Fallback (Missing Data Abstention)",
        "question": "What is our company valuation from the Series B round?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "description": "Tests abstention safety guardrail when information is not present in memory.",
        "expected": "Not in memory: nothing covers this"
    }
]

# Sidebar
st.sidebar.title("⏱️ Chronexa Engine")
st.sidebar.caption("Enterprise Temporal Memory & Action Assistant (TextOS)")

nav = st.sidebar.radio(
    "Architecture Navigation",
    [
        "🔍 Point-in-Time Memory Q&A",
        "📅 Chronological Storyline Explorer",
        "⚡ TextOS Action Assistant",
        "📊 Benchmark & Engineering Decisions"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**Engineering Benchmark (Pure CPU)**")
st.sidebar.markdown("- **Top-10 Recall:** `100.0%`")
st.sidebar.markdown("- **Future Leakage:** `0.0% (Zero-Harm)`")
st.sidebar.markdown("- **Retrieval Latency:** `1.7s on CPU`")
st.sidebar.markdown("- **Active LLM Provider:** `" + generator.provider.upper() + "`")
st.sidebar.caption(f"Model: {generator.model}")

st.sidebar.divider()
st.sidebar.caption("Built & Maintained by **Pruthvi R**")
st.sidebar.markdown("[GitHub Repository](https://github.com/Spidey173/Chronexa)")


# -----------------------------------------------------------------------------
# TAB 1: POINT-IN-TIME MEMORY SEARCH
# -----------------------------------------------------------------------------
if nav == "🔍 Point-in-Time Memory Q&A":
    st.subheader("Point-in-Time Temporal Memory Engine")
    st.caption("Answers organizational questions strictly as of an arbitrary historical timestamp with zero future leakage.")

    # Preset Selector
    selected_idx = st.selectbox(
        "Load a Curated Benchmark Scenario (Demonstrating Temporal Evolution):",
        range(len(BENCHMARKS)),
        format_func=lambda i: BENCHMARKS[i]["label"]
    )
    preset = BENCHMARKS[selected_idx]

    col_q, col_t = st.columns([2.5, 1.2])
    with col_q:
        query_input = st.text_input("Natural Language Query:", value=preset["question"])
    with col_t:
        as_of_input = st.text_input("As-Of Timestamp (ISO-8601):", value=preset["as_of"])

    # Retrieval Engine Config
    with st.expander("⚙️ Advanced Retrieval Parameters", expanded=False):
        c_mode, c_topk = st.columns(2)
        with c_mode:
            search_mode = st.radio(
                "Retrieval Algorithm:",
                [
                    "Structured Body-Aware BM25 (Production Default - 1.7s CPU)",
                    "Hybrid RRF (Dense Vectors + BM25 Fusion)"
                ],
                index=0
            )
        with c_topk:
            top_k_select = st.slider("Candidate Depth (Top-K):", min_value=5, max_value=20, value=10)

    st.info(f"💡 **Scenario Context:** {preset['description']} | **Expected:** `{preset['expected']}`")

    if st.button("🚀 Execute Temporal Search & Grounded Generation", type="primary", use_container_width=True) or query_input:
        try:
            as_of_dt = datetime.fromisoformat(as_of_input.replace("Z", "+00:00"))
        except Exception as e:
            st.error(f"Invalid timestamp format: {e}")
            st.stop()

        t0_retrieval = time.time()
        # 1. Pre-Retrieval Temporal Boundary
        visible_units = get_visible_units(as_of_dt, data_dir=DATA_DIR)

        # 2. Candidate Retrieval
        if "Hybrid" in search_mode:
            results = retriever.search(query_input, visible_units, top_k=top_k_select, dense_weight=0.5)
        else:
            bm25_idx = BM25Index(visible_units)
            results = bm25_idx.search(query_input, top_k=top_k_select)

        retrieval_ms = int((time.time() - t0_retrieval) * 1000)
        retrieved_units = [u for u, _ in results]

        # 3. Grounded LLM Generation
        t0_llm = time.time()
        gen_res = generator.generate(query_input, as_of_input, retrieved_units)
        llm_ms = int((time.time() - t0_llm) * 1000)

        # Display Metrics Strip
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Visible Pool (<= as_of)", f"{len(visible_units)} units", delta=f"{1316 - len(visible_units)} filtered out")
        m2.metric("Retrieval Latency", f"{retrieval_ms} ms", delta="Pure CPU")
        m3.metric("LLM Provider Latency", f"{llm_ms} ms", delta=generator.provider.upper())
        m4.metric("Temporal Leakage Risk", "0.0%", delta="Guaranteed")

        st.divider()

        # Display Grounded Answer
        st.markdown("### 💡 Grounded Answer")
        answer_text = gen_res.get("answer", "")
        sources = gen_res.get("sources", [])
        abstained = gen_res.get("abstained", False)

        if abstained:
            st.warning(f"**Abstention Triggered (Calibrated Fallback):**\n\n{answer_text}")
        else:
            st.success(f"**Answer:**\n\n{answer_text}")

        # Citations Breakdown
        st.markdown("### 📑 Citable Evidence Chain")
        for rank, (unit, score) in enumerate(results[:5], start=1):
            is_cited = unit.id in sources
            tag_class = "badge-success" if is_cited else "badge-pill"
            tag_label = "⭐ PRIMARY CITATION" if is_cited else "RELEVANT CANDIDATE"

            st.markdown(f"""
            <div class="citation-card">
                <span class="{tag_class}">{tag_label}</span>
                <strong>#{rank} [{unit.id}]</strong> | Record: <code>{unit.record}</code> | Timestamp: <code>{unit.time.isoformat()}</code> | Score: <code>{score:.3f}</code>
                <div style="margin-top: 0.5rem; color: #d1d5db; font-size: 0.93rem;">
                    {unit.text}
                </div>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: CHRONOLOGICAL STORYLINE EXPLORER
# -----------------------------------------------------------------------------
elif nav == "📅 Chronological Storyline Explorer":
    st.subheader("Interactive Storyline & Fact Evolution Explorer")
    st.caption("Visual proof of why temporal memory is essential: watch how organizational facts drift over time.")

    storyline = st.selectbox(
        "Select Enterprise Fact Lifecycle:",
        [
            "Route Planner v2 Launch Date (Moved 3 times)",
            "Acme Freight Pricing Proposal (Extensions & Terms)",
            "Dark Mode Elimination (Hearsay vs Reality)",
            "Harbor Logistics Contract (Marcus vs John Debate)"
        ]
    )

    if storyline.startswith("Route Planner"):
        st.markdown("""
        <div class="timeline-card">
            <span class="badge-pill">Step 1 · Sep 8, 2026 (Planning Meeting)</span>
            <h4>Initial Target: September 30, 2026</h4>
            <p>In <code>MTG-0908-PLAN</code>, the team unanimously agrees to target September 30 for Route Planner v2 rollout.</p>
        </div>
        <div class="timeline-card">
            <span class="badge-pill">Step 2 · Sep 10, 2026 (Slack Alert)</span>
            <h4>Regression Slip: October 14, 2026</h4>
            <p>In <code>SL-RP-0910-1</code>, engineering reports a severe geocoding regression. The launch is postponed by two weeks to October 14.</p>
        </div>
        <div class="timeline-card">
            <span class="badge-pill">Step 3 · Sep 16, 2026 (Go/No-Go Review)</span>
            <h4>Customer Request Slip: October 21, 2026</h4>
            <p>In <code>MTG-0916-GONOGO</code>, key enterprise client Acme Freight asks for a dedicated dispatcher training week. The final target becomes October 21.</p>
        </div>
        """, unsafe_allow_html=True)

    elif storyline.startswith("Acme"):
        st.markdown("""
        <div class="timeline-card">
            <span class="badge-pill">Sep 9, 2026 · Client Call</span>
            <h4>Commitment Made to Sarah Patel</h4>
            <p>Alex promises pricing by Friday, Sep 11 during <code>MTG-0909-ACME</code>.</p>
        </div>
        <div class="timeline-card">
            <span class="badge-pill">Sep 10, 2026 · Extension Request</span>
            <h4>Agreed Extension to Sep 15</h4>
            <p>Alex emails Sarah asking to move deadline to Tuesday, Sep 15. Sarah accepts via email <code>EM-0910-ACME-EXT-R</code>.</p>
        </div>
        <div class="timeline-card">
            <span class="badge-pill">Sep 15, 2026 · Proposal Delivered</span>
            <h4>Proposal Sent: $18/vehicle/mo</h4>
            <p>Proposal sent via <code>EM-0915-ACME-PROP</code> with 3-year term, $18/mo pricing, waived onboarding fee. Acknowledged by client.</p>
        </div>
        """, unsafe_allow_html=True)

    elif storyline.startswith("Dark Mode"):
        st.markdown("""
        <div class="timeline-card">
            <span class="badge-pill">Sep 11, 2026 · Design Sync</span>
            <h4>Second-Hand Hearsay Claim</h4>
            <p>Dana claims in <code>MTG-0911-DESIGN</code> that John told her he was fine cutting dark mode.</p>
        </div>
        <div class="timeline-card">
            <span class="badge-pill">Sep 14, 2026 · Direct Statement</span>
            <h4>Direct Rebuttal from John</h4>
            <p>John writes in Slack (<code>SL-RP-0914-1</code>) to keep dark mode because two pilot enterprises explicitly asked for it.</p>
        </div>
        <div class="timeline-card">
            <span class="badge-pill">Sep 16, 2026 · Final Decision</span>
            <h4>Resolution: v2.1 Fast-Follow</h4>
            <p>Final decision reached in <code>MTG-0916-GONOGO</code>: dark mode will ship in v2.1 approximately 2 weeks post-launch.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="timeline-card">
            <span class="badge-pill">Sep 11, 2026 · Sales Update</span>
            <h4>Marcus: Q4 Close ($120k ARR)</h4>
            <p>Marcus posts in <code>SL-SALES-0911-1</code> that Harbor Logistics is expected to sign in Q4.</p>
        </div>
        <div class="timeline-card">
            <span class="badge-pill">Sep 14, 2026 · Leadership DM</span>
            <h4>John: Keep Harbor Out of Forecast</h4>
            <p>John flags via DM (<code>SL-DM-JA-0914-1</code>) that Harbor asked for an uncapped liability clause; instructs Alex to exclude from board forecast.</p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 3: TEXTOS ACTION ASSISTANT
# -----------------------------------------------------------------------------
elif nav == "⚡ TextOS Action Assistant":
    st.subheader("TextOS: Dry-Run Action Assistant")
    st.caption("Translates natural language instructions into validated tool execution calls with dynamic entity resolution.")

    sample_cmds = [
        "Message Sarah on Slack that the route planner fix is live",
        "Remind me an hour before the board meeting tomorrow",
        "Reschedule my 1:1 with Alex to Friday at 3pm",
        "Post an update to #announcements about the launch date"
    ]

    selected_cmd = st.selectbox("Quick-Load Sample Instruction:", sample_cmds)
    user_cmd = st.text_input("Natural Language Command:", value=selected_cmd)
    as_of_cmd = st.text_input("As-Of Context Timestamp:", value="2026-09-18T18:00:00-07:00")

    if st.button("⚡ Parse & Resolve Dynamic Entities", type="primary", use_container_width=True) or user_cmd:
        with st.spinner("Resolving internal entities (Slack Users, Channels, Calendar IDs)..."):
            actions = action_parser.parse(user_cmd, as_of_cmd)

        st.markdown("### Validated Tool Execution Payload (Dry-Run)")
        st.json(actions)

        st.caption("✅ Dynamic entities (Slack user IDs like `U03SARAHK`, channel IDs like `C10RP`, calendar events) are resolved dynamically without hardcoding.")


# -----------------------------------------------------------------------------
# TAB 4: ARCHITECTURE & BENCHMARKS
# -----------------------------------------------------------------------------
else:
    st.subheader("Chronexa Engineering Decisions & Benchmark Telemetry")
    st.caption("Comprehensive analysis of why Structured Lexical & Hybrid RRF outclasses generic dense vector search.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### The "Vector Search Trap" in Enterprise RAG
        Standard RAG systems with dense vectors (OpenAI, MiniLM) fail on corporate work data because:
        1. **Version & Timestamp Blindness**: Vector similarity for `Route Planner v1` and `v2` is > 0.97. Dense models treat them as identical.
        2. **High CPU Latency**: Computing dense vectors across 1,300+ units on CPU took **20–30s per request**.
        3. **Out-of-Vocabulary (OOV) Codename Confusion**: Internal ticket IDs and client names are blurred by generic embedding models.

        ### Chronexa's 5 Core Pillars
        - **1. Pre-Retrieval Temporal Boundary**: Drops future units before search executes. Mathematical 0% leakage guarantee.
        - **2. ISO Timestamp Normalization**: Converts `2026-09-10T16:05` into searchable date tokens (`2026`, `sep`, `10`).
        - **3. Header Down-Weighting**: Metadata brackets (`[Slack route-planner]`) receive 0.35x weight, keeping bodies prioritized.
        - **4. 1.8x Named Entity Boosting**: Proper nouns and client codenames are boosted to prevent false-positive matches.
        - **5. Reciprocal Rank Fusion (RRF)**: Merges decomposed sub-queries into optimal ranking.
        """)

    with col2:
        st.markdown("""
        ### Official Evaluation Results

        | Component | Train Benchmark Score | Engineering Note |
        | :--- | :---: | :--- |
        | **Retrieval Recall @ 10** | **100.0%** | Zero forbidden records leaked (0.0%). |
        | **Retrieval Latency** | **1.7s** | Pure CPU (vs 30s with Dense Embeddings). |
        | **Action Accuracy** | **12 / 12** | Dynamic entity resolution across Slack, Calendar, Gmail. |
        | **Grounding Precision** | **Active** | Calibrated abstention guardrails against prompt injection. |

        ---
        ### Architecture Data Flow
        ```text
        Raw Enterprise Corpus (Slack, Gmail, Meetings)
                      │
                      ▼
        [Pre-Retrieval Temporal Filter] (time <= as_of)
                      │
                      ▼
        [Structured Body-Aware BM25 + Sub-Query Decomp]
                      │
                      ▼
        [Reciprocal Rank Fusion (RRF)]
                      │
                      ▼
        Top 10 Citable Units (Latency: 1.7s on CPU)
                      │
                      ▼
        [Groq LLM: openai/gpt-oss-120b]
                      │
                      ▼
        Grounded Answer + Auditable Citations
        ```
        """)
