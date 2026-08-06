"""Ranking metrics for retrieval evaluation."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    hits = sum(1 for r in relevant if r in top)
    return hits / len(relevant)


def precision_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(1 for r in top if r in rel) / len(top)


def mrr(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    rel = set(relevant)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    total = 0.0
    for i, gain in enumerate(gains[:k], start=1):
        total += gain / math.log2(i + 1)
    return total


def ndcg_at_k(relevance_grades: Sequence[float], k: int) -> float:
    """relevance_grades is aligned with retrieved order (grade for each retrieved doc)."""
    dcg = dcg_at_k(relevance_grades, k)
    ideal = dcg_at_k(sorted(relevance_grades, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
