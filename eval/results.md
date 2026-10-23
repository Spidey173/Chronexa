# Chronexa Retrieval Evaluation Results Benchmarks

| Split | Query Set | Config | N | Recall@5 | Recall@10 | Recall@20 | MRR | Found None@20 | Forbidden@10 | Latency/Q | Ans Strict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| holdout | paraphrases | `hybrid_rrf_rerank` | 80 | 70.6% | 86.8% | 88.2% | 0.6037 | 5.9% | 0 | 2309ms | N/A |
| holdout | original | `hybrid_rrf_rerank` | 20 | 70.6% | 76.5% | 88.2% | 0.7591 | 0.0% | 0 | 2688ms | N/A |
| holdout | paraphrases | `hybrid_rrf` | 80 | 72.1% | 83.8% | 88.2% | 0.5955 | 5.9% | 0 | 1712ms | N/A |
| holdout | original | `hybrid_rrf` | 20 | 70.6% | 76.5% | 88.2% | 0.7297 | 0.0% | 0 | 1648ms | N/A |
