"""Run BM25 retrieval evaluation against the local CloudSearch API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metrics import mean, mrr, precision_at_k, recall_at_k  # noqa: E402


def load_dataset() -> list[dict]:
    path = ROOT / "datasets" / "retrieval" / "phase1.json"
    return json.loads(path.read_text())


def search(api: str, query: str, limit: int = 10) -> list[str]:
    resp = requests.get(f"{api}/v1/search", params={"q": query, "limit": limit}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [hit["document_id"] for hit in data.get("results", [])]


def main() -> None:
    api = os.getenv("CLOUDSEARCH_API_URL", "http://localhost:8080").rstrip("/")
    k = int(os.getenv("EVAL_K", "5"))
    dataset = load_dataset()

    recalls, precisions, mrrs = [], [], []
    print(f"Evaluating BM25 against {api} (K={k})")
    print("-" * 60)

    for item in dataset:
        retrieved = search(api, item["query"], limit=max(k, 10))
        relevant = item["relevant_document_ids"]
        r = recall_at_k(relevant, retrieved, k)
        p = precision_at_k(relevant, retrieved, k)
        m = mrr(relevant, retrieved)
        recalls.append(r)
        precisions.append(p)
        mrrs.append(m)
        print(f"{item['id']}: recall@{k}={r:.2f} precision@{k}={p:.2f} mrr={m:.2f} retrieved={retrieved[:k]}")

    print("-" * 60)
    summary = {
        "strategy": "bm25",
        f"recall@{k}": round(mean(recalls), 4),
        f"precision@{k}": round(mean(precisions), 4),
        "mrr": round(mean(mrrs), 4),
        "n": len(dataset),
    }
    print(json.dumps(summary, indent=2))

    out = ROOT / "experiments" / "bm25_phase1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
