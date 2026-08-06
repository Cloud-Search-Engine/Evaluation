# CloudSearch Evaluation

Benchmark datasets and metrics for retrieval / RAG quality. This repo answers: **does search and grounding actually work?** It is separate from the production Backend.

Sibling repos: **Backend** (API under test), **Ingestion** (corpus), **Database** (schema).

## What’s in this repo

| Path | Purpose |
| --- | --- |
| `datasets/retrieval/` | Query → relevant `document_id`s (Phase 1 JSON) |
| `datasets/rag/` | Questions with expected concepts / forbidden claims |
| `datasets/architecture/` | Cross-cloud architecture advisor questions |
| `metrics/` | Recall@K, Precision@K, MRR, NDCG, faithfulness helpers |
| `benchmarks/bm25/` | Runner against live `/v1/search` |
| `benchmarks/vector|hybrid|reranking/` | Placeholders for later phases |
| `experiments/` | Saved run summaries (gitignored except `.gitkeep`) |
| `requirements.txt` | Python deps |

## Prerequisites

- Python **3.11+**
- Running Backend API with an ingested corpus (`CLOUDSEARCH_API_URL`)

## How to start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Unit-test metrics
python -m pytest metrics -q

# Run BM25 benchmark against local API
export CLOUDSEARCH_API_URL=http://localhost:8080
export EVAL_K=5
python -m benchmarks.bm25.run
```

Results are written to `experiments/bm25_phase1.json`.

## Full local stack before evaluating

```bash
# from parent Cloud_Search_Engine folder
docker compose up --build -d
docker compose --profile seed run --rm crawler
```

## Interpreting metrics

| Metric | Meaning |
| --- | --- |
| Recall@K | Fraction of relevant docs found in top K |
| Precision@K | Fraction of top K that are relevant |
| MRR | Reciprocal rank of first relevant hit |
| NDCG@K | Ranking quality with graded relevance |

Only publish measured numbers — do not use placeholder percentages on a resume.
