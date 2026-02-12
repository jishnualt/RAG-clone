"""Basic evaluation scaffold for RAG responses."""
from typing import List, Dict


def evaluate_responses(pairs: List[Dict[str, str]]) -> List[Dict[str, float]]:
    """Placeholder evaluation using simple word overlap.

    Args:
        pairs: list of dicts with keys `question`, `answer`, `reference`.
    Returns:
        List of dictionaries with a naive `overlap` score.
    """
    results = []
    for p in pairs:
        answer_words = set(str(p.get("answer", "")).lower().split())
        ref_words = set(str(p.get("reference", "")).lower().split())
        overlap = 0.0
        if ref_words:
            overlap = len(answer_words & ref_words) / len(ref_words)
        results.append({"overlap": overlap})
    return results
