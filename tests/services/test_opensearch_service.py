from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from app.models.opensearch import IndexDocumentRequest
from app.services.config import OpenSearchConfig
from app.services.document_service import DocumentService
from app.services.opensearch_service import OpenSearchService, OpenSearchServiceError


# ---------------------------------------------------------------------------
# Fake OpenSearch client (mirrors FakeS3Client pattern)
# ---------------------------------------------------------------------------


class FakeOpenSearchClient:
    """Minimal stand-in for ``opensearchpy.OpenSearch``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

        # Configurable responses / exceptions
        self.search_response: dict[str, Any] | Exception = {"hits": {"total": {"value": 0}, "hits": []}}
        self.bulk_response: dict[str, Any] | Exception = {"errors": False, "items": []}
        self.count_response: dict[str, Any] | Exception = {"count": 0}
        self.delete_response: dict[str, Any] | Exception = {"result": "deleted"}
        self.delete_by_query_response: dict[str, Any] | Exception = {"deleted": 0}

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("search", kwargs))
        if isinstance(self.search_response, Exception):
            raise self.search_response
        return self.search_response

    def bulk(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("bulk", kwargs))
        if isinstance(self.bulk_response, Exception):
            raise self.bulk_response
        return self.bulk_response

    def count(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("count", kwargs))
        if isinstance(self.count_response, Exception):
            raise self.count_response
        return self.count_response

    def delete(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("delete", kwargs))
        if isinstance(self.delete_response, Exception):
            raise self.delete_response
        return self.delete_response

    def delete_by_query(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("delete_by_query", kwargs))
        if isinstance(self.delete_by_query_response, Exception):
            raise self.delete_by_query_response
        return self.delete_by_query_response


# ---------------------------------------------------------------------------
# Stub DocumentService (returns predictable chunks)
# ---------------------------------------------------------------------------


class StubDocumentService(DocumentService):
    """DocumentService that skips Bedrock and returns canned chunk results."""

    def __init__(self, chunks: list[str] | None = None) -> None:
        # Bypass parent __init__ (which reads env vars for Bedrock).
        self._embedding_model_id = ""
        self._embedding_dim = 1024
        self._chunks = chunks if chunks is not None else ["chunk-0", "chunk-1"]

    def chunk_text(self, *, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
        if not text or not text.strip():
            return []
        return list(self._chunks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = OpenSearchConfig(
    endpoint="https://fake.us-west-2.aoss.amazonaws.com",
    index_name="test-index",
    region="us-west-2",
)


def _make_service(
    fake: FakeOpenSearchClient,
    stub_doc: StubDocumentService | None = None,
) -> OpenSearchService:
    doc = stub_doc or StubDocumentService()
    service = OpenSearchService(config=_DEFAULT_CONFIG, document_service=doc)
    service._client = lambda: fake  # type: ignore[method-assign]
    return service


# ===========================================================================
# document_exists
# ===========================================================================


def test_document_exists_returns_true_when_hits_found():
    fake = FakeOpenSearchClient()
    fake.search_response = {"hits": {"total": {"value": 3}, "hits": []}}
    service = _make_service(fake)

    assert asyncio.run(service.document_exists(filename="a.md")) is True
    assert fake.calls[0][0] == "search"


def test_document_exists_returns_false_when_no_hits():
    fake = FakeOpenSearchClient()
    fake.search_response = {"hits": {"total": {"value": 0}, "hits": []}}
    service = _make_service(fake)

    assert asyncio.run(service.document_exists(filename="missing.md")) is False


def test_document_exists_raises_on_blank_filename():
    fake = FakeOpenSearchClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="filename must be provided"):
        asyncio.run(service.document_exists(filename=""))

    with pytest.raises(ValueError, match="filename must be provided"):
        asyncio.run(service.document_exists(filename="   "))


def test_document_exists_wraps_exceptions():
    fake = FakeOpenSearchClient()
    fake.search_response = RuntimeError("connection refused")
    service = _make_service(fake)

    with pytest.raises(OpenSearchServiceError, match="Failed to check document existence"):
        asyncio.run(service.document_exists(filename="a.md"))


# ===========================================================================
# list_indexed_documents
# ===========================================================================


def test_list_indexed_documents_returns_sorted_filenames():
    fake = FakeOpenSearchClient()
    fake.search_response = {
        "hits": {"total": {"value": 0}, "hits": []},
        "aggregations": {
            "unique_filenames": {
                "buckets": [
                    {"key": "z-doc.md", "doc_count": 3},
                    {"key": "a-doc.md", "doc_count": 1},
                ]
            }
        },
    }
    service = _make_service(fake)

    result = asyncio.run(service.list_indexed_documents())
    assert result == ["a-doc.md", "z-doc.md"]


def test_list_indexed_documents_returns_empty_on_empty_index():
    fake = FakeOpenSearchClient()
    fake.search_response = {
        "hits": {"total": {"value": 0}, "hits": []},
        "aggregations": {"unique_filenames": {"buckets": []}},
    }
    service = _make_service(fake)

    assert asyncio.run(service.list_indexed_documents()) == []


def test_list_indexed_documents_wraps_exceptions():
    fake = FakeOpenSearchClient()
    fake.search_response = RuntimeError("timeout")
    service = _make_service(fake)

    with pytest.raises(OpenSearchServiceError, match="Failed to list indexed documents"):
        asyncio.run(service.list_indexed_documents())


# ===========================================================================
# index_document
# ===========================================================================


def test_index_document_indexes_chunks():
    fake = FakeOpenSearchClient()
    # document_exists will use search → no hits
    fake.search_response = {"hits": {"total": {"value": 0}, "hits": []}}
    fake.bulk_response = {
        "errors": False,
        "items": [
            {"index": {"_id": "id-0", "status": 201}},
            {"index": {"_id": "id-1", "status": 201}},
        ],
    }
    service = _make_service(fake)

    resp = asyncio.run(service.index_document(filename="doc.md", content="some text"))
    assert resp.filename == "doc.md"
    assert resp.chunk_count == 2
    assert resp.doc_ids == ["id-0", "id-1"]
    assert resp.skipped is False

    # Verify both search (exists check) and bulk (indexing) were called
    call_names = [c[0] for c in fake.calls]
    assert "search" in call_names
    assert "bulk" in call_names


def test_index_document_skips_when_already_indexed():
    fake = FakeOpenSearchClient()
    fake.search_response = {"hits": {"total": {"value": 5}, "hits": []}}
    service = _make_service(fake)

    resp = asyncio.run(service.index_document(filename="existing.md", content="text"))
    assert resp.skipped is True
    assert resp.chunk_count == 0
    assert resp.doc_ids == []

    # Only search was called (no bulk)
    call_names = [c[0] for c in fake.calls]
    assert "search" in call_names
    assert "bulk" not in call_names


def test_index_document_returns_zero_chunks_on_empty_content_after_strip():
    fake = FakeOpenSearchClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="content must be provided"):
        asyncio.run(service.index_document(filename="doc.md", content=""))


def test_index_document_raises_on_blank_filename():
    fake = FakeOpenSearchClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="filename must be provided"):
        asyncio.run(service.index_document(filename="", content="text"))


def test_index_document_empty_chunks_returns_zero():
    """When chunk_text returns no chunks (e.g. very short text), we get 0 doc_ids."""
    fake = FakeOpenSearchClient()
    fake.search_response = {"hits": {"total": {"value": 0}, "hits": []}}
    stub_doc = StubDocumentService(chunks=[])
    service = _make_service(fake, stub_doc)

    resp = asyncio.run(service.index_document(filename="tiny.md", content="x"))
    assert resp.chunk_count == 0
    assert resp.doc_ids == []
    assert resp.skipped is False


# ===========================================================================
# bulk_index_documents
# ===========================================================================


def test_bulk_index_documents_mixes_new_and_existing():
    """Tests mixed scenario with parallel execution: one doc exists, one is new."""

    fake = FakeOpenSearchClient()
    lock = threading.Lock()

    # Thread-safe side-effect: match by filename in the query body.
    def _search_side_effect(**kwargs: Any) -> dict[str, Any]:
        body = kwargs.get("body", {})
        filename = body.get("query", {}).get("term", {}).get("filename", "")
        with lock:
            fake.calls.append(("search", kwargs))
        if filename == "old.md":
            return {"hits": {"total": {"value": 3}, "hits": []}}
        return {"hits": {"total": {"value": 0}, "hits": []}}

    fake.search = _search_side_effect  # type: ignore[method-assign]

    fake.bulk_response = {
        "errors": False,
        "items": [{"index": {"_id": "new-1", "status": 201}}],
    }

    stub_doc = StubDocumentService(chunks=["only-chunk"])
    service = _make_service(fake, stub_doc)

    docs = [
        IndexDocumentRequest(filename="old.md", content="already indexed"),
        IndexDocumentRequest(filename="new.md", content="new content"),
    ]
    resp = asyncio.run(service.bulk_index_documents(documents=docs))

    assert resp.indexed_count == 1
    assert resp.skipped_count == 1
    assert resp.total_chunks == 1
    assert len(resp.results) == 2


def test_bulk_index_documents_all_skipped():
    """All documents already exist — everything should be skipped."""
    fake = FakeOpenSearchClient()
    fake.search_response = {"hits": {"total": {"value": 1}, "hits": []}}
    service = _make_service(fake)

    docs = [
        IndexDocumentRequest(filename="a.md", content="aaa"),
        IndexDocumentRequest(filename="b.md", content="bbb"),
    ]
    resp = asyncio.run(service.bulk_index_documents(documents=docs))

    assert resp.indexed_count == 0
    assert resp.skipped_count == 2
    assert resp.total_chunks == 0


def test_bulk_index_documents_empty_list():
    """Empty document list should return zero counts."""
    fake = FakeOpenSearchClient()
    service = _make_service(fake)

    resp = asyncio.run(service.bulk_index_documents(documents=[]))

    assert resp.indexed_count == 0
    assert resp.skipped_count == 0
    assert resp.total_chunks == 0
    assert resp.results == []


def test_bulk_index_documents_respects_max_concurrency():
    """Verify the semaphore limits concurrent execution."""
    fake = FakeOpenSearchClient()
    fake.search_response = {"hits": {"total": {"value": 0}, "hits": []}}
    fake.bulk_response = {
        "errors": False,
        "items": [{"index": {"_id": "id-0", "status": 201}}],
    }

    peak = {"current": 0, "max": 0}
    lock = threading.Lock()
    original_search = fake.search

    def _tracking_search(**kwargs: Any) -> dict[str, Any]:
        with lock:
            peak["current"] += 1
            if peak["current"] > peak["max"]:
                peak["max"] = peak["current"]
        result = original_search(**kwargs)
        with lock:
            peak["current"] -= 1
        return result

    fake.search = _tracking_search  # type: ignore[method-assign]

    stub_doc = StubDocumentService(chunks=["chunk"])
    service = _make_service(fake, stub_doc)

    docs = [IndexDocumentRequest(filename=f"doc-{i}.md", content=f"content {i}") for i in range(10)]
    resp = asyncio.run(service.bulk_index_documents(documents=docs, max_concurrency=2))

    assert resp.indexed_count == 10
    # Peak concurrent searches should not exceed max_concurrency
    assert peak["max"] <= 2


# ===========================================================================
# search
# ===========================================================================


def test_search_hybrid_returns_hits():
    fake = FakeOpenSearchClient()
    fake.search_response = {
        "hits": {
            "total": {"value": 1},
            "hits": [
                {
                    "_id": "hit-1",
                    "_score": 0.85,
                    "_source": {"filename": "doc.md", "content": "found text"},
                }
            ],
        }
    }
    service = _make_service(fake)

    resp = asyncio.run(service.search(query="test"))
    assert resp.search_type == "hybrid"
    assert resp.total_hits == 1
    assert resp.hits[0].doc_id == "hit-1"
    assert resp.hits[0].score == 0.85
    assert resp.hits[0].filename == "doc.md"

    # Check the search_pipeline param was passed
    _, kwargs = fake.calls[0]
    assert kwargs.get("params", {}).get("search_pipeline") == "sagemaker-docs-search-pipeline"


def test_search_text_does_not_use_pipeline():
    fake = FakeOpenSearchClient()
    fake.search_response = {"hits": {"total": {"value": 0}, "hits": []}}
    service = _make_service(fake)

    resp = asyncio.run(service.search(query="test", search_type="text"))
    assert resp.search_type == "text"
    _, kwargs = fake.calls[0]
    assert kwargs.get("params", {}) == {}


def test_search_vector_does_not_use_pipeline():
    fake = FakeOpenSearchClient()
    fake.search_response = {"hits": {"total": {"value": 0}, "hits": []}}
    service = _make_service(fake)

    asyncio.run(service.search(query="test", search_type="vector"))
    _, kwargs = fake.calls[0]
    assert kwargs.get("params", {}) == {}


def test_search_raises_on_blank_query():
    fake = FakeOpenSearchClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="query must be provided"):
        asyncio.run(service.search(query=""))


def test_search_empty_results():
    fake = FakeOpenSearchClient()
    fake.search_response = {"hits": {"total": {"value": 0}, "hits": []}}
    service = _make_service(fake)

    resp = asyncio.run(service.search(query="nothing"))
    assert resp.total_hits == 0
    assert resp.hits == []


def test_search_wraps_exceptions():
    fake = FakeOpenSearchClient()
    fake.search_response = RuntimeError("timeout")
    service = _make_service(fake)

    with pytest.raises(OpenSearchServiceError, match="Failed to execute search"):
        asyncio.run(service.search(query="boom"))


# ===========================================================================
# delete_document
# ===========================================================================


def test_delete_document_success():
    fake = FakeOpenSearchClient()
    fake.delete_response = {"result": "deleted"}
    service = _make_service(fake)

    assert asyncio.run(service.delete_document(doc_id="abc")) is True


def test_delete_document_not_found():
    fake = FakeOpenSearchClient()

    class NotFoundError(Exception):
        pass

    NotFoundError.__name__ = "NotFoundError"
    fake.delete_response = NotFoundError("404")
    service = _make_service(fake)

    assert asyncio.run(service.delete_document(doc_id="missing")) is False


def test_delete_document_raises_on_blank_id():
    fake = FakeOpenSearchClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="doc_id must be provided"):
        asyncio.run(service.delete_document(doc_id=""))


def test_delete_document_wraps_unexpected_error():
    fake = FakeOpenSearchClient()
    fake.delete_response = RuntimeError("server error")
    service = _make_service(fake)

    with pytest.raises(OpenSearchServiceError, match="Failed to delete document"):
        asyncio.run(service.delete_document(doc_id="abc"))


# ===========================================================================
# delete_documents_by_filename
# ===========================================================================


def test_delete_documents_by_filename_returns_count():
    fake = FakeOpenSearchClient()
    fake.delete_by_query_response = {"deleted": 4}
    service = _make_service(fake)

    assert asyncio.run(service.delete_documents_by_filename(filename="doc.md")) == 4


def test_delete_documents_by_filename_raises_on_blank():
    fake = FakeOpenSearchClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="filename must be provided"):
        asyncio.run(service.delete_documents_by_filename(filename=""))


def test_delete_documents_by_filename_wraps_errors():
    fake = FakeOpenSearchClient()
    fake.delete_by_query_response = RuntimeError("fail")
    service = _make_service(fake)

    with pytest.raises(OpenSearchServiceError, match="Failed to delete documents"):
        asyncio.run(service.delete_documents_by_filename(filename="x.md"))


# ===========================================================================
# get_index_stats
# ===========================================================================


def test_get_index_stats_returns_count():
    fake = FakeOpenSearchClient()
    fake.count_response = {"count": 42}
    service = _make_service(fake)

    resp = asyncio.run(service.get_index_stats())
    assert resp.index_name == "test-index"
    assert resp.doc_count == 42
    assert resp.status == "available"


def test_get_index_stats_wraps_errors():
    fake = FakeOpenSearchClient()
    fake.count_response = RuntimeError("down")
    service = _make_service(fake)

    with pytest.raises(OpenSearchServiceError, match="Failed to get index stats"):
        asyncio.run(service.get_index_stats())


# ===========================================================================
# _bulk_index_chunks error handling
# ===========================================================================


def test_bulk_index_chunks_raises_on_partial_errors():
    fake = FakeOpenSearchClient()
    fake.search_response = {"hits": {"total": {"value": 0}, "hits": []}}
    fake.bulk_response = {
        "errors": True,
        "items": [
            {"index": {"_id": "ok", "status": 201}},
            {"index": {"_id": "fail", "status": 400, "error": {"type": "mapper_parsing_exception"}}},
        ],
    }
    service = _make_service(fake)

    with pytest.raises(OpenSearchServiceError, match="Bulk indexing had"):
        asyncio.run(service.index_document(filename="bad.md", content="text"))
