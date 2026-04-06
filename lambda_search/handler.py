"""AWS Lambda handler for FAISS + BM25 hybrid search.

Loads search artifacts (FAISS index, BM25 corpus, corpus metadata) from S3
on cold start and caches them in module-level globals for warm invocations.

Supported actions (passed via ``event["action"]``):

- ``search`` — hybrid, text-only, or vector-only search
- ``index_documents`` — rebuild indexes with new documents
- ``document_exists`` — check if a filename has indexed chunks
- ``list_documents`` — list all unique indexed filenames
- ``delete_document`` — remove a single chunk by doc_id
- ``delete_by_filename`` — remove all chunks for a filename
- ``get_stats`` — return index statistics

Environment variables:

- ``SEARCH_INDEX_BUCKET`` — S3 bucket containing index artifacts
- ``SEARCH_INDEX_PREFIX`` — S3 key prefix (default ``search-index/``)
- ``BEDROCK_EMBEDDING_MODEL_ID`` — Bedrock model for embeddings
- ``BEDROCK_EMBEDDING_DIM`` — embedding vector dimension
- ``BM25_WEIGHT`` — weight for BM25 scores in hybrid fusion
- ``VECTOR_WEIGHT`` — weight for vector scores in hybrid fusion
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
import uuid
from typing import Any

import boto3
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

S3_BUCKET = os.environ.get("SEARCH_INDEX_BUCKET", "")
S3_PREFIX = os.environ.get("SEARCH_INDEX_PREFIX", "search-index/")
EMBEDDING_MODEL_ID = os.environ.get("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBEDDING_DIM = int(os.environ.get("BEDROCK_EMBEDDING_DIM", "1024"))
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "0.3"))
VECTOR_WEIGHT = float(os.environ.get("VECTOR_WEIGHT", "0.7"))

# ---------------------------------------------------------------------------
# Module-level cache (persists across warm Lambda invocations)
# ---------------------------------------------------------------------------

_faiss_index: faiss.Index | None = None
_bm25_index: BM25Okapi | None = None
_corpus: list[dict[str, str]] | None = None  # [{doc_id, filename, content}, ...]
_artifacts_loaded: bool = False

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

_s3 = boto3.client("s3")
_bedrock = boto3.client("bedrock-runtime")


def _s3_key(name: str) -> str:
    return f"{S3_PREFIX}{name}"


def _load_artifacts() -> None:
    """Download and deserialize FAISS index, BM25 corpus, and metadata from S3."""

    global _faiss_index, _bm25_index, _corpus, _artifacts_loaded

    if _artifacts_loaded:
        return

    logger.info("Loading search artifacts from s3://%s/%s", S3_BUCKET, S3_PREFIX)

    tmp = tempfile.gettempdir()

    # FAISS index
    faiss_path = os.path.join(tmp, "faiss.index")
    _s3.download_file(S3_BUCKET, _s3_key("faiss.index"), faiss_path)
    _faiss_index = faiss.read_index(faiss_path)

    # Corpus metadata
    corpus_obj = _s3.get_object(Bucket=S3_BUCKET, Key=_s3_key("corpus.pkl"))
    _corpus = pickle.loads(corpus_obj["Body"].read())

    # BM25 index
    bm25_obj = _s3.get_object(Bucket=S3_BUCKET, Key=_s3_key("bm25.pkl"))
    _bm25_index = pickle.loads(bm25_obj["Body"].read())

    _artifacts_loaded = True
    logger.info("Artifacts loaded: %d documents in corpus", len(_corpus))


def _save_artifacts() -> None:
    """Serialize and upload current in-memory artifacts to S3."""

    if _faiss_index is None or _corpus is None or _bm25_index is None:
        raise RuntimeError("Cannot save artifacts: indexes not built")

    tmp = tempfile.gettempdir()

    # FAISS index
    faiss_path = os.path.join(tmp, "faiss.index")
    faiss.write_index(_faiss_index, faiss_path)
    _s3.upload_file(faiss_path, S3_BUCKET, _s3_key("faiss.index"))

    # Corpus metadata
    corpus_bytes = pickle.dumps(_corpus)
    _s3.put_object(Bucket=S3_BUCKET, Key=_s3_key("corpus.pkl"), Body=corpus_bytes)

    # BM25 index
    bm25_bytes = pickle.dumps(_bm25_index)
    _s3.put_object(Bucket=S3_BUCKET, Key=_s3_key("bm25.pkl"), Body=bm25_bytes)

    logger.info("Artifacts saved to s3://%s/%s", S3_BUCKET, S3_PREFIX)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _embed_text(text: str) -> np.ndarray:
    """Generate an embedding vector for *text* via Bedrock Titan."""

    response = _bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
    )
    body = json.loads(response["body"].read())
    vec = np.array(body["embedding"], dtype=np.float32)
    # L2-normalize for cosine similarity via inner product
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed multiple texts, returning a (N, dim) array."""

    vectors = [_embed_text(t) for t in texts]
    return np.vstack(vectors)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def _rebuild_indexes() -> None:
    """Rebuild FAISS and BM25 indexes from current corpus."""

    global _faiss_index, _bm25_index

    if not _corpus:
        _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        _bm25_index = BM25Okapi([])
        return

    # Re-embed all corpus content
    texts = [doc["content"] for doc in _corpus]
    vectors = _embed_texts(texts)

    _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    _faiss_index.add(vectors)

    tokenized = [text.lower().split() for text in texts]
    _bm25_index = BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range via min-max scaling."""

    smin = scores.min()
    smax = scores.max()
    if smax - smin < 1e-9:
        return np.zeros_like(scores)
    return (scores - smin) / (smax - smin)


def _fuse_scores(
    bm25_scores: np.ndarray,
    vector_scores: np.ndarray,
    size: int,
) -> list[tuple[int, float]]:
    """Fuse BM25 and vector scores using weighted arithmetic mean.

    Returns:
        List of (corpus_index, fused_score) tuples sorted by score descending.
    """

    bm25_norm = _min_max_normalize(bm25_scores)
    vector_norm = _min_max_normalize(vector_scores)
    fused = BM25_WEIGHT * bm25_norm + VECTOR_WEIGHT * vector_norm

    top_indices = np.argsort(fused)[::-1][:size]
    return [(int(idx), float(fused[idx])) for idx in top_indices if fused[idx] > 0]


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


def _handle_search(event: dict[str, Any]) -> dict[str, Any]:
    _load_artifacts()
    assert _faiss_index is not None and _bm25_index is not None and _corpus is not None

    query = event["query"]
    size = event.get("size", 10)
    search_type = event.get("search_type", "hybrid")

    if not _corpus:
        return {"query": query, "search_type": search_type, "total_hits": 0, "hits": []}

    hits: list[dict[str, Any]] = []

    if search_type == "text":
        tokenized_query = query.lower().split()
        scores = _bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:size]
        for idx in top_indices:
            if scores[idx] > 0:
                doc = _corpus[idx]
                hits.append({
                    "doc_id": doc["doc_id"],
                    "score": float(scores[idx]),
                    "filename": doc["filename"],
                    "content": doc["content"],
                })

    elif search_type == "vector":
        query_vec = _embed_text(query).reshape(1, -1)
        distances, indices = _faiss_index.search(query_vec, min(size, _faiss_index.ntotal))
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and distances[0][i] > 0:
                doc = _corpus[idx]
                hits.append({
                    "doc_id": doc["doc_id"],
                    "score": float(distances[0][i]),
                    "filename": doc["filename"],
                    "content": doc["content"],
                })

    else:  # hybrid
        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = _bm25_index.get_scores(tokenized_query)

        # Vector scores
        query_vec = _embed_text(query).reshape(1, -1)
        n_docs = _faiss_index.ntotal
        distances, indices = _faiss_index.search(query_vec, n_docs)
        vector_scores = np.zeros(len(_corpus))
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                vector_scores[idx] = distances[0][i]

        # Fuse
        fused = _fuse_scores(bm25_scores, vector_scores, size)
        for idx, score in fused:
            doc = _corpus[idx]
            hits.append({
                "doc_id": doc["doc_id"],
                "score": score,
                "filename": doc["filename"],
                "content": doc["content"],
            })

    return {
        "query": query,
        "search_type": search_type,
        "total_hits": len(hits),
        "hits": hits,
    }


def _handle_index_documents(event: dict[str, Any]) -> dict[str, Any]:
    """Index documents into the search index.

    Each document can provide either:
    - ``chunks``: list of pre-chunked strings (backward compatible)
    - ``content``: raw text that will be chunked by Lambda

    Optional ``skip_existing`` (default False) skips documents whose
    filename is already in the corpus.
    """

    global _corpus, _artifacts_loaded

    _load_artifacts()
    if _corpus is None:
        _corpus = []

    skip_existing = event.get("skip_existing", False)
    existing_filenames = {doc["filename"] for doc in _corpus} if skip_existing else set()

    documents = event["documents"]
    all_doc_ids: list[str] = []
    skipped_filenames: list[str] = []

    for doc in documents:
        filename = doc["filename"]

        if skip_existing and filename in existing_filenames:
            skipped_filenames.append(filename)
            continue

        # Determine chunks: use pre-chunked if provided, otherwise chunk raw content
        if "chunks" in doc:
            chunks = doc["chunks"]
        elif "content" in doc:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            chunk_size = event.get("chunk_size", 500)
            chunk_overlap = event.get("chunk_overlap", 50)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_text(doc["content"])
        else:
            continue

        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            _corpus.append({"doc_id": doc_id, "filename": filename, "content": chunk})
            all_doc_ids.append(doc_id)

        existing_filenames.add(filename)

    if all_doc_ids:
        _rebuild_indexes()
        _save_artifacts()

    return {"doc_ids": all_doc_ids, "skipped_filenames": skipped_filenames}


def _handle_document_exists(event: dict[str, Any]) -> dict[str, Any]:
    _load_artifacts()
    assert _corpus is not None

    filename = event["filename"]
    exists = any(doc["filename"] == filename for doc in _corpus)
    return {"exists": exists}


def _handle_list_documents(event: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    _load_artifacts()
    assert _corpus is not None

    filenames = sorted(set(doc["filename"] for doc in _corpus))
    return {"filenames": filenames}


def _handle_delete_document(event: dict[str, Any]) -> dict[str, Any]:
    global _corpus

    _load_artifacts()
    assert _corpus is not None

    doc_id = event["doc_id"]
    original_len = len(_corpus)
    _corpus = [doc for doc in _corpus if doc["doc_id"] != doc_id]
    deleted = len(_corpus) < original_len

    if deleted:
        _rebuild_indexes()
        _save_artifacts()

    return {"deleted": deleted}


def _handle_delete_by_filename(event: dict[str, Any]) -> dict[str, Any]:
    global _corpus

    _load_artifacts()
    assert _corpus is not None

    filename = event["filename"]
    original_len = len(_corpus)
    _corpus = [doc for doc in _corpus if doc["filename"] != filename]
    deleted_count = original_len - len(_corpus)

    if deleted_count > 0:
        _rebuild_indexes()
        _save_artifacts()

    return {"deleted_count": deleted_count}


def _handle_build_index(event: dict[str, Any]) -> dict[str, Any]:
    """Build index from raw documents: chunk, embed, index, and save.

    Accepts a list of raw documents and handles the full pipeline:
    chunking, embedding, FAISS+BM25 index building, and S3 persistence.

    Event fields:
        documents: list of {filename, content} dicts
        chunk_size: int (default 500)
        chunk_overlap: int (default 50)
        skip_existing: bool (default True)
    """

    global _corpus, _faiss_index, _bm25_index, _artifacts_loaded

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = event["documents"]
    chunk_size = event.get("chunk_size", 500)
    chunk_overlap = event.get("chunk_overlap", 50)
    skip_existing = event.get("skip_existing", True)

    # Try loading existing artifacts; initialize empty if none exist
    try:
        _load_artifacts()
    except Exception:
        logger.info("No existing artifacts found — initializing empty index")
        _corpus = []
        _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        _bm25_index = BM25Okapi([])
        _artifacts_loaded = True

    if _corpus is None:
        _corpus = []

    existing_filenames = {doc["filename"] for doc in _corpus}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )

    processed_count = 0
    skipped_count = 0
    total_chunks_added = 0

    for doc in documents:
        filename = doc["filename"]
        content = doc["content"]

        if skip_existing and filename in existing_filenames:
            skipped_count += 1
            continue

        chunks = splitter.split_text(content)
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            _corpus.append({"doc_id": doc_id, "filename": filename, "content": chunk})
            total_chunks_added += 1

        existing_filenames.add(filename)
        processed_count += 1

    if total_chunks_added > 0:
        _rebuild_indexes()
        _save_artifacts()

    return {
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "total_chunks_added": total_chunks_added,
    }


def _handle_get_stats(event: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    _load_artifacts()
    assert _corpus is not None

    return {
        "index_name": "sagemaker-docs",
        "doc_count": len(_corpus),
        "status": "available",
    }


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

_ACTION_MAP = {
    "search": _handle_search,
    "index_documents": _handle_index_documents,
    "build_index": _handle_build_index,
    "document_exists": _handle_document_exists,
    "list_documents": _handle_list_documents,
    "delete_document": _handle_delete_document,
    "delete_by_filename": _handle_delete_by_filename,
    "get_stats": _handle_get_stats,
}


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ARG001
    """AWS Lambda entry point.

    Args:
        event: JSON payload with ``action`` field plus action-specific fields.
        context: Lambda context (unused).

    Returns:
        JSON-serializable response dict.
    """

    action = event.get("action")
    if not action:
        return {"error": "Missing 'action' field in event"}

    handler = _ACTION_MAP.get(action)
    if handler is None:
        return {"error": f"Unknown action: {action}"}

    try:
        return handler(event)
    except Exception as exc:
        logger.exception("Action '%s' failed", action)
        return {"error": str(exc)}
