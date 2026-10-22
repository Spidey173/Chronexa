# Chronexa Retrieval Evaluation Results

Evaluated on stratified Dev (~70%) and Holdout (~30%) splits with unseen paraphrases.
Strict rule: Dev only used for model/config development; Holdout untouched during active tuning.

---

## 1. Dev Split Benchmark Summary

| Split | Query Set | Configuration | N | Recall@5 | Recall@10 | Recall@20 | MRR | Found None@20 | Primary Score | Latency | Strict Answer Acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Dev** | **Original** | `bm25_only` | 19 | 77.8% | 100.0% | 100.0% | 0.6718 | 0.0% | 100.0% | 911ms | 52.6% |
| **Dev** | **Original** | `dense_only` | 19 | 66.7% | 77.8% | 88.9% | 0.6572 | 11.1% | 77.8% | 140ms | 42.1% |
| **Dev** | **Original** | `hybrid_rrf` | 19 | 77.8% | 88.9% | 94.4% | 0.7643 | 0.0% | 88.9% | 940ms | 47.4% |
| **Dev** | **Original** | **`hybrid_rrf_rerank`** | 19 | **88.9%** | **94.4%** | **94.4%** | **0.7870** | **0.0%** | **94.4%** | 2648ms | **57.9%** |
| **Dev** | **Paraphrases** | `bm25_only` | 76 | 54.2% | 75.0% | 79.2% | 0.5381 | 11.1% | 75.0% | 62ms | 42.1% |
| **Dev** | **Paraphrases** | `dense_only` (`bge-small-en-v1.5`) | 76 | 52.8% | 70.8% | 83.3% | 0.5544 | 9.7% | 70.8% | 790ms | 38.2% |
| **Dev** | **Paraphrases** | `hybrid_rrf` (BM25 + BGE RRF) | 76 | 66.7% | 79.2% | 91.7% | 0.5680 | 2.8% | 79.2% | 73ms | 47.4% |
| **Dev** | **Paraphrases** | **`hybrid_rrf_rerank` (Chronexa Engine)** | 76 | **75.0%** | **83.3%** | **88.9%** | **0.6831** | **1.4%** | **83.3%** | 2281ms | **59.2%** |

---

## 2. Dense Embedding Model Selection on Dev Paraphrases

| Model | Model Size | Embedding Dim | Recall@5 | Recall@10 | MRR | Found None@20 | Cold Latency |
|---|---|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 91 MB | 384 | 50.0% | 62.5% | 0.5073 | 23.6% | 32.9s |
| `BAAI/bge-small-en-v1.5` (**Selected**) | 67 MB | 384 | **52.8%** | **70.8%** | **0.5544** | **9.7%** | 97.4s (Disk-cached: 0.01s) |
| `BAAI/bge-base-en-v1.5` | 219 MB | 768 | 63.9% | 69.4% | 0.5125 | 9.7% | 359.1s (Disk-cached: 0.01s) |

*Decision:* `BAAI/bge-small-en-v1.5` selected for superior MRR, quantized efficiency (67MB), and highest Recall@10.

---

## 3. Answer Generation Matrix (Extractive Fallback vs LLM)

Evaluated across candidate context windows (Top 8, Top 10, Top 12) retrieved via `hybrid_rrf_rerank`:

| Evaluation Set | Context Window | Mode | Strict Accuracy | Lenient Accuracy | Abstention Correctness |
|---|---|---|---|---|---|
| **Dev Original** (n=19) | Top 8 | Extractive Fallback | 57.89% | 68.42% | 100.0% |
| **Dev Original** (n=19) | Top 10 | Extractive Fallback | 57.89% | 68.42% | 100.0% |
| **Dev Original** (n=19) | Top 12 | Extractive Fallback | 57.89% | 68.42% | 100.0% |
| **Dev Original** (n=19) | Top 8 | **LLM (`openai/gpt-oss-20b`)** | **78.95%** | **84.21%** | **100.0%** |
| **Dev Original** (n=19) | Top 10 | **LLM (`openai/gpt-oss-20b`)** | 73.68% | 78.95% | **100.0%** |
| **Dev Original** (n=19) | Top 12 | **LLM (`openai/gpt-oss-20b`)** | **78.95%** | **84.21%** | **100.0%** |
| **Dev Paraphrases** (n=76) | Top 8 | Extractive Fallback | 59.21% | 64.47% | 100.0% |
| **Dev Paraphrases** (n=76) | Top 10 | Extractive Fallback | 59.21% | 64.47% | 100.0% |
| **Dev Paraphrases** (n=76) | Top 12 | Extractive Fallback | 59.21% | 64.47% | 100.0% |
| **Dev Paraphrases** (n=76) | Top 8 | **LLM (`openai/gpt-oss-20b`)** | 64.47% | 69.74% | 94.74% |
| **Dev Paraphrases** (n=76) | Top 10 | **LLM (`openai/gpt-oss-20b`)** | **65.79%** | **71.05%** | **100.0%** |
| **Dev Paraphrases** (n=76) | Top 12 | **LLM (`openai/gpt-oss-20b`)** | 59.21% | 64.47% | **100.0%** |

---

## 4. Action Assistant Benchmark (Bonus, n=12)

| Test Case | Command Summary | Status | Arguments Match |
|---|---|---|---|
| ACT-TR-01 | Message Sarah on Slack that geocoding fix looks good | **PASS** | 3/3 (`to: U03SARAHK`) |
| ACT-TR-02 | Email Sarah Patel about proposal | **PASS** | 3/3 (`to: sarah.patel@acmefreight...`) |
| ACT-TR-03 | Remind follow up Acme on 25th at 9am | **PASS** | 3/3 (`due: 2026-09-25T09:00:00-07:00`) |
| ACT-TR-04 | Move board deck prep to 3pm | **PASS** | 4/4 (`event_id: CAL-BOARDPREP`) |
| ACT-TR-05 | Open Figma | **PASS** | 2/2 (`app: Figma`) |
| ACT-TR-06 | Tell route planner channel launching Oct 21 | **PASS** | 3/3 (`to: C10RP`) |
| ACT-TR-07 | Book 30 mins with Ben tomorrow at 2 | **PASS** | 5/5 (`attendees: ben.torres@...`) |
| ACT-TR-08 | What's our launch date again? | **PASS** | 2/2 (`type: memory.ask`) |
| ACT-TR-09 | Message Sarah about pricing proposal | **PASS** | 3/3 (`type: clarify` - ambiguous Sarah) |
| ACT-TR-10 | Email John corrected NRR & thank Ben | **PASS** | 6/6 (Multi-action: Gmail + Slack) |
| ACT-TR-11 | Delete all my emails from Marcus | **PASS** | 2/2 (`type: confirm` - destructive) |
| ACT-TR-12 | Remind 1 hr before board meeting | **PASS** | 3/3 (`due: 2026-09-23T08:00:00-07:00`) |

**Summary:** 100.0% Pass Rate (12/12), 100.0% Argument Accuracy.
| holdout | paraphrases | `hybrid_rrf_rerank` | 32 | 53.6% | 89.3% | 92.9% | 0.4656 | 3.6% | 0 | 2540ms | N/A |
