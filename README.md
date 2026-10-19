# ⏱️ Chronexa: Work Memory Search & Action Assistant

A desktop work-assistant project that searches across scattered workplace communications (Slack messages, Gmail, Google Calendar events, and meeting transcripts) and executes dry-run actions safely.

---

## 💡 Why I Built This

At any tech company, information lives in different silos: Slack chats, emails, Google Calendar, and meeting notes. When you ask a question like *"When is Route Planner launching?"* or *"Did I send the proposal to Sarah?"*, basic search tools often fail because:

1. **Information changes over time:** A launch date might move from Sept 30 to Oct 14, then to Oct 21. If you ask what the plan was on Sept 10, the assistant shouldn't cheat by looking at an email sent on Sept 16.
2. **Keywords vs. Meaning:** Searching for *"pricing agreement"* might miss an email that says *"contract rates"*. Pure keyword search misses synonyms, while pure vector search misses exact dates and IDs.
3. **Safety in Actions:** If an assistant translates *"Delete all emails from Marcus"* into an action, it should ask for confirmation before doing anything destructive.

Chronexa was built to solve these three practical challenges cleanly using Python.

---

## 🚀 Key Features

* **📅 Point-in-Time Date Filtering:** Filters the data strictly before retrieval (`timestamp <= as_of`). If you ask a question as of Sept 10, it physically ignores all records created after Sept 10.
* **🔍 Hybrid Search (BM25 + Embeddings):** 
  * Uses **Okapi BM25** for exact matching (IDs, names, dates).
  * Uses **FastEmbed (`bge-small-en-v1.5`)** for semantic matching (synonyms, paraphrased questions).
  * Merges both ranking lists using **Reciprocal Rank Fusion (RRF)** and reranks top candidates.
* **🤖 Action Assistant (TextOS):** Translates natural language commands into structured actions:
  * `slack.send_message` (resolves names to Slack IDs)
  * `gmail.send` (resolves contact emails and drafts body)
  * `calendar.update_event` (calculates ISO timestamps with timezone offsets)
  * **Safety:** Ambiguous names trigger a `clarify` question; destructive tasks trigger a `confirm` prompt.
* **💻 Interactive Streamlit Demo:** A simple dark-mode web UI to test queries, adjust the historical date slider, and inspect retrieved records.
* **⚡ 100% Local & CPU-Friendly:** Runs completely offline on a standard laptop CPU without requiring a GPU or paid cloud API.

---

## 📊 Evaluation & Results

I evaluated the system on a benchmark dataset of 27 memory questions and 12 action commands:

| Metric | Result | What it Means |
|---|:---:|---|
| **Retrieval Coverage** | **100%** | The system found all required source documents in the top 10 results. |
| **Future Leakage** | **0%** | Zero records from the future were shown when answering historical questions. |
| **CPU Search Speed** | **< 1.7s** | Fast enough for live interactive search on a regular laptop. |
| **Action Assistant Accuracy** | **12 / 12 (100%)** | All 12 test commands were parsed into the correct tool calls with exact arguments. |
| **Unseen Paraphrases** | **+20.8% Recall** | Adding embeddings and reranking helped retrieve answers even when questions were rephrased. |

---

## ⚡ How to Run Locally

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Spidey173/Chronexa.git
cd Chronexa

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Benchmark Tests
```bash
# Runs retrieval evaluation, deterministic answer checks, and action assistant
./run.sh all
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501` to test queries with the historical date slider.

---

## 🛠️ How It Works (Step-by-Step)

```text
User Question + Historical Date (as_of)
                 │
                 ▼
1. Date Filter: Remove any record where timestamp > as_of
                 │
                 ▼
2. Dual Search:
   ├── Lexical Search (BM25 for exact terms and dates)
   └── Dense Search (FastEmbed vectors for semantic meaning)
                 │
                 ▼
3. Rank Fusion (RRF): Combine scores so top items from both methods rise to the top
                 │
                 ▼
4. Answer / Action:
   ├── Grounded Answer: Summarize facts strictly from retrieved sources
   └── Action Parser: Output structured JSON for Slack / Gmail / Calendar
```

---

## 📁 Project Structure

```text
Chronexa/
├── data/                  # Mock data: Slack, Gmail, Calendar, Meetings, Dictations
├── src/
│   ├── ingestion/         # Loads records and filters them by date
│   ├── retrieval/         # BM25 keyword search, dense embeddings, and RRF ranker
│   ├── generation/        # Formats prompt and generates answers
│   └── actions/           # Parses commands into Slack/Email/Calendar actions
├── eval_harness/          # Scripts to test retrieval accuracy and action parsing
├── evals/                 # Benchmark question and command datasets
├── streamlit_app.py       # Interactive Streamlit demo dashboard
├── requirements.txt       # Dependencies (numpy, fastembed, streamlit, dateparser)
└── run.sh                 # Single script to run all tests
```

---

## 💬 What I Learned Building This

1. **Why naive RAG leaks data:** If you just dump all documents into a vector database, the model will answer past questions with future knowledge. Pre-filtering data by timestamp before searching completely eliminates this bug.
2. **Why hybrid search matters:** BM25 alone fails when users use informal synonyms; vector embeddings alone fail on exact serial IDs and dates. Combining both gives the best of both worlds.
3. **Action safety:** AI agents should never directly execute destructive actions (like mass email deletion) without a human-in-the-loop confirmation step.

---

## 👤 Author

**Pruthvi R**  
* **GitHub:** [@Spidey173](https://github.com/Spidey173)  
* **Portfolio:** [pruthvi-17.vercel.app](https://pruthvi-17.vercel.app/)  
* **LinkedIn:** [linkedin.com/in/pruthvir173](https://www.linkedin.com/in/pruthvir173/)
