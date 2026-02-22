"""Shared utilities for RAG search experiment notebooks."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

BASE_URL = "http://localhost:8000"


def search(query: str, search_type: str = "hybrid", size: int = 10) -> dict[str, Any]:
    """Call the FastAPI search endpoint and return the parsed response.

    Args:
        query: The search query text.
        search_type: One of "hybrid", "text", or "vector".
        size: Maximum number of results to return.

    Returns:
        Parsed JSON response matching the SearchResponse schema.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status.
    """
    resp = requests.post(
        f"{BASE_URL}/opensearch/search",
        json={"query": query, "size": size, "search_type": search_type},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def results_to_dataframe(response: dict[str, Any]) -> pd.DataFrame:
    """Convert a SearchResponse dict to a pandas DataFrame.

    Args:
        response: Parsed JSON from the search endpoint.

    Returns:
        DataFrame with columns: rank, doc_id, score, filename, content_preview.
    """
    rows = []
    for i, hit in enumerate(response.get("hits", []), start=1):
        rows.append({
            "rank": i,
            "doc_id": hit["doc_id"],
            "score": round(hit["score"], 4),
            "filename": hit["filename"],
            "content_preview": hit["content"][:120] + "..." if len(hit["content"]) > 120 else hit["content"],
        })
    return pd.DataFrame(rows)


def hits_to_doc_ids(response: dict[str, Any]) -> list[str]:
    """Extract ordered doc_id list from a SearchResponse.

    Args:
        response: Parsed JSON from the search endpoint.

    Returns:
        List of doc_id strings in rank order.
    """
    return [hit["doc_id"] for hit in response.get("hits", [])]


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets.

    Args:
        set_a: First set of identifiers.
        set_b: Second set of identifiers.

    Returns:
        Jaccard index (0.0 to 1.0). Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def overlap_matrix(
    results_dict: dict[str, list[str]],
) -> pd.DataFrame:
    """Build a Jaccard similarity matrix from multiple result lists.

    Args:
        results_dict: Mapping of label -> list of doc_ids.
            Example: {"hybrid": [...], "text": [...], "vector": [...]}.

    Returns:
        Square DataFrame with Jaccard similarities between all pairs.
    """
    labels = list(results_dict.keys())
    sets = {label: set(ids) for label, ids in results_dict.items()}
    matrix: dict[str, dict[str, float]] = {}
    for a in labels:
        matrix[a] = {}
        for b in labels:
            matrix[a][b] = round(jaccard_similarity(sets[a], sets[b]), 3)
    return pd.DataFrame(matrix, index=labels)


def rank_biased_overlap(list_a: list[str], list_b: list[str], p: float = 0.9) -> float:
    """Compute rank-biased overlap (RBO) between two ranked lists.

    A position-weighted similarity metric where top ranks contribute more
    than lower ranks. The parameter *p* controls the top-heaviness:
    lower values weight the top more heavily.

    Reference: Webber, Moffat, Zobel (2010).

    Args:
        list_a: First ranked list of identifiers.
        list_b: Second ranked list of identifiers.
        p: Persistence parameter (0 < p < 1). Default 0.9.

    Returns:
        RBO score (0.0 to 1.0).
    """
    if not list_a or not list_b:
        return 0.0

    min_len = min(len(list_a), len(list_b))
    rbo_sum = 0.0

    for d in range(1, min_len + 1):
        set_a = set(list_a[:d])
        set_b = set(list_b[:d])
        agreement = len(set_a & set_b) / d
        rbo_sum += (p ** (d - 1)) * agreement

    return (1 - p) * rbo_sum
