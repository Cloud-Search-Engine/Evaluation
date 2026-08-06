"""Lightweight faithfulness helpers for RAG evaluation (Phase 4+)."""

from __future__ import annotations

from typing import Sequence


def concept_coverage(answer: str, expected_concepts: Sequence[str]) -> float:
    if not expected_concepts:
        return 0.0
    lower = answer.lower()
    hits = sum(1 for c in expected_concepts if c.lower() in lower)
    return hits / len(expected_concepts)


def contains_forbidden(answer: str, forbidden_claims: Sequence[str]) -> bool:
    lower = answer.lower()
    return any(f.lower() in lower for f in forbidden_claims)
