from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import pytest

from app.models.search import IndexDocumentRequest
from app.services.config import SearchConfig
from app.services.document_service import DocumentService
from app.services.search_service import SearchService, SearchServiceError


# ---------------------------------------------------------------------------
# Fake Lambda client (mirrors FakeOpenSearchClient pattern)
# ---------------------------------------------------------------------------


class FakeLambdaClient:
    """Minimal stand-in for ``boto3.client("lambda")``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._response_data: dict[str, Any] = {}
        self._function_error: str | None = None

    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("invoke", kwargs))

        payload_data = json.dumps(self._response_data).encode()
        resp: dict[str, Any] = {
            "StatusCode": 200,
            "Payload": io.BytesIO(payload_data),
        }
        if self._function_error:
            resp["FunctionError"] = self._function_error
        return resp

    def set_response(self, data: dict[str, Any]) -> None:
        self._response_data = data
        self._function_error = None

    def set_error(self, error_message: str) -> None:
        self._response_data = {"errorMessage": error_message}
        self._function_error = "Unhandled"


# ---------------------------------------------------------------------------
# Stub DocumentService (returns predictable chunks)
# ---------------------------------------------------------------------------


class StubDocumentService(DocumentService):
    """DocumentService that skips Bedrock and returns canned chunk results."""

    def __init__(self, chunks: list[str] | None = None) -> None:
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

_DEFAULT_CONFIG = SearchConfig(
    lambda_function_name="test-search-lambda",
    index_bucket="test-bucket",
    index_prefix="test-index/",
    region="us-east-1",
)


def _make_service(
    fake: FakeLambdaClient,
    stub_doc: StubDocumentService | None = None,
) -> SearchService:
    doc = stub_doc or StubDocumentService()
    service = SearchService(config=_DEFAULT_CONFIG, document_service=doc)
    service._client = lambda: fake  # type: ignore[method-assign]
    return service


# ===========================================================================
# document_exists
# ===========================================================================


def test_document_exists_returns_true():
    fake = FakeLambdaClient()
    fake.set_response({"exists": True})
    service = _make_service(fake)

    assert asyncio.run(service.document_exists(filename="a.md")) is True
    assert len(fake.calls) == 1
    payload = json.loads(fake.calls[0][1]["Payload"])
    assert payload["action"] == "document_exists"
    assert payload["filename"] == "a.md"


def test_document_exists_returns_false():
    fake = FakeLambdaClient()
    fake.set_response({"exists": False})
    service = _make_service(fake)

    assert asyncio.run(service.document_exists(filename="missing.md")) is False


def test_document_exists_raises_on_blank_filename():
    fake = FakeLambdaClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="filename must be provided"):
        asyncio.run(service.document_exists(filename=""))

    with pytest.raises(ValueError, match="filename must be provided"):
        asyncio.run(service.document_exists(filename="   "))


def test_document_exists_wraps_lambda_error():
    fake = FakeLambdaClient()
    fake.set_error("connection refused")
    service = _make_service(fake)

    with pytest.raises(SearchServiceError, match="Lambda execution failed"):
        asyncio.run(service.document_exists(filename="a.md"))


# ===========================================================================
# list_indexed_documents
# ===========================================================================


def test_list_indexed_documents_returns_filenames():
    fake = FakeLambdaClient()
    fake.set_response({"filenames": ["a-doc.md", "z-doc.md"]})
    service = _make_service(fake)

    result = asyncio.run(service.list_indexed_documents())
    assert result == ["a-doc.md", "z-doc.md"]


def test_list_indexed_documents_returns_empty():
    fake = FakeLambdaClient()
    fake.set_response({"filenames": []})
    service = _make_service(fake)

    assert asyncio.run(service.list_indexed_documents()) == []


def test_list_indexed_documents_wraps_errors():
    fake = FakeLambdaClient()
    fake.set_error("timeout")
    service = _make_service(fake)

    with pytest.raises(SearchServiceError, match="Lambda execution failed"):
        asyncio.run(service.list_indexed_documents())


# ===========================================================================
# index_document
# ===========================================================================


def test_index_document_indexes_chunks():
    fake = FakeLambdaClient()
    call_count = {"n": 0}
    original_invoke = fake.invoke

    def _invoke(**kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        payload = json.loads(kwargs["Payload"])
        if payload["action"] == "document_exists":
            fake.set_response({"exists": False})
        elif payload["action"] == "index_documents":
            fake.set_response({"doc_ids": ["id-0", "id-1"]})
        return original_invoke(**kwargs)

    fake.invoke = _invoke  # type: ignore[method-assign]
    service = _make_service(fake)

    resp = asyncio.run(service.index_document(filename="doc.md", content="some text"))
    assert resp.filename == "doc.md"
    assert resp.chunk_count == 2
    assert resp.doc_ids == ["id-0", "id-1"]
    assert resp.skipped is False


def test_index_document_skips_when_already_indexed():
    fake = FakeLambdaClient()
    fake.set_response({"exists": True})
    service = _make_service(fake)

    resp = asyncio.run(service.index_document(filename="existing.md", content="text"))
    assert resp.skipped is True
    assert resp.chunk_count == 0
    assert resp.doc_ids == []
    # Only document_exists was called
    assert len(fake.calls) == 1


def test_index_document_raises_on_blank_filename():
    fake = FakeLambdaClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="filename must be provided"):
        asyncio.run(service.index_document(filename="", content="text"))


def test_index_document_raises_on_blank_content():
    fake = FakeLambdaClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="content must be provided"):
        asyncio.run(service.index_document(filename="doc.md", content=""))


def test_index_document_empty_chunks_returns_zero():
    fake = FakeLambdaClient()
    fake.set_response({"exists": False})
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
    fake = FakeLambdaClient()
    original_invoke = fake.invoke

    def _invoke(**kwargs: Any) -> dict[str, Any]:
        payload = json.loads(kwargs["Payload"])
        if payload["action"] == "document_exists":
            if payload["filename"] == "old.md":
                fake.set_response({"exists": True})
            else:
                fake.set_response({"exists": False})
        elif payload["action"] == "index_documents":
            fake.set_response({"doc_ids": ["new-1"]})
        return original_invoke(**kwargs)

    fake.invoke = _invoke  # type: ignore[method-assign]
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
    fake = FakeLambdaClient()
    fake.set_response({"exists": True})
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
    fake = FakeLambdaClient()
    service = _make_service(fake)

    resp = asyncio.run(service.bulk_index_documents(documents=[]))

    assert resp.indexed_count == 0
    assert resp.skipped_count == 0
    assert resp.total_chunks == 0
    assert resp.results == []


# ===========================================================================
# search
# ===========================================================================


def test_search_hybrid_returns_hits():
    fake = FakeLambdaClient()
    fake.set_response({
        "total_hits": 1,
        "hits": [
            {"doc_id": "hit-1", "score": 0.85, "filename": "doc.md", "content": "found text"},
        ],
    })
    service = _make_service(fake)

    resp = asyncio.run(service.search(query="test"))
    assert resp.search_type == "hybrid"
    assert resp.total_hits == 1
    assert resp.hits[0].doc_id == "hit-1"
    assert resp.hits[0].score == 0.85
    assert resp.hits[0].filename == "doc.md"

    # Verify search_type was sent to Lambda
    payload = json.loads(fake.calls[0][1]["Payload"])
    assert payload["search_type"] == "hybrid"


def test_search_text_type():
    fake = FakeLambdaClient()
    fake.set_response({"total_hits": 0, "hits": []})
    service = _make_service(fake)

    resp = asyncio.run(service.search(query="test", search_type="text"))
    assert resp.search_type == "text"
    payload = json.loads(fake.calls[0][1]["Payload"])
    assert payload["search_type"] == "text"


def test_search_vector_type():
    fake = FakeLambdaClient()
    fake.set_response({"total_hits": 0, "hits": []})
    service = _make_service(fake)

    asyncio.run(service.search(query="test", search_type="vector"))
    payload = json.loads(fake.calls[0][1]["Payload"])
    assert payload["search_type"] == "vector"


def test_search_raises_on_blank_query():
    fake = FakeLambdaClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="query must be provided"):
        asyncio.run(service.search(query=""))


def test_search_empty_results():
    fake = FakeLambdaClient()
    fake.set_response({"total_hits": 0, "hits": []})
    service = _make_service(fake)

    resp = asyncio.run(service.search(query="nothing"))
    assert resp.total_hits == 0
    assert resp.hits == []


def test_search_wraps_lambda_error():
    fake = FakeLambdaClient()
    fake.set_error("timeout")
    service = _make_service(fake)

    with pytest.raises(SearchServiceError, match="Lambda execution failed"):
        asyncio.run(service.search(query="boom"))


# ===========================================================================
# delete_document
# ===========================================================================


def test_delete_document_success():
    fake = FakeLambdaClient()
    fake.set_response({"deleted": True})
    service = _make_service(fake)

    assert asyncio.run(service.delete_document(doc_id="abc")) is True


def test_delete_document_not_found():
    fake = FakeLambdaClient()
    fake.set_response({"deleted": False})
    service = _make_service(fake)

    assert asyncio.run(service.delete_document(doc_id="missing")) is False


def test_delete_document_raises_on_blank_id():
    fake = FakeLambdaClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="doc_id must be provided"):
        asyncio.run(service.delete_document(doc_id=""))


def test_delete_document_wraps_lambda_error():
    fake = FakeLambdaClient()
    fake.set_error("server error")
    service = _make_service(fake)

    with pytest.raises(SearchServiceError, match="Lambda execution failed"):
        asyncio.run(service.delete_document(doc_id="abc"))


# ===========================================================================
# delete_documents_by_filename
# ===========================================================================


def test_delete_documents_by_filename_returns_count():
    fake = FakeLambdaClient()
    fake.set_response({"deleted_count": 4})
    service = _make_service(fake)

    assert asyncio.run(service.delete_documents_by_filename(filename="doc.md")) == 4


def test_delete_documents_by_filename_raises_on_blank():
    fake = FakeLambdaClient()
    service = _make_service(fake)

    with pytest.raises(ValueError, match="filename must be provided"):
        asyncio.run(service.delete_documents_by_filename(filename=""))


def test_delete_documents_by_filename_wraps_errors():
    fake = FakeLambdaClient()
    fake.set_error("fail")
    service = _make_service(fake)

    with pytest.raises(SearchServiceError, match="Lambda execution failed"):
        asyncio.run(service.delete_documents_by_filename(filename="x.md"))


# ===========================================================================
# get_index_stats
# ===========================================================================


def test_get_index_stats_returns_stats():
    fake = FakeLambdaClient()
    fake.set_response({"index_name": "sagemaker-docs", "doc_count": 42, "status": "available"})
    service = _make_service(fake)

    resp = asyncio.run(service.get_index_stats())
    assert resp.index_name == "sagemaker-docs"
    assert resp.doc_count == 42
    assert resp.status == "available"


def test_get_index_stats_wraps_errors():
    fake = FakeLambdaClient()
    fake.set_error("down")
    service = _make_service(fake)

    with pytest.raises(SearchServiceError, match="Lambda execution failed"):
        asyncio.run(service.get_index_stats())


# ===========================================================================
# _invoke_lambda error handling
# ===========================================================================


def test_invoke_lambda_raises_on_error_field():
    fake = FakeLambdaClient()
    fake.set_response({"error": "Unknown action: bad"})
    service = _make_service(fake)

    with pytest.raises(SearchServiceError, match="Search operation failed"):
        asyncio.run(service.search(query="test"))


# ===========================================================================
# build_index
# ===========================================================================


def test_build_index_sends_correct_payload():
    fake = FakeLambdaClient()
    fake.set_response({"processed_count": 2, "skipped_count": 0, "total_chunks_added": 10})
    service = _make_service(fake)

    docs = [
        {"filename": "a.md", "content": "content a"},
        {"filename": "b.md", "content": "content b"},
    ]
    result = asyncio.run(service.build_index(documents=docs))

    assert result["processed_count"] == 2
    assert result["total_chunks_added"] == 10

    assert len(fake.calls) == 1
    payload = json.loads(fake.calls[0][1]["Payload"])
    assert payload["action"] == "build_index"
    assert len(payload["documents"]) == 2
    assert payload["chunk_size"] == 500
    assert payload["chunk_overlap"] == 50
    assert payload["skip_existing"] is True


def test_build_index_batching():
    """With batch_size=2 and 5 docs, expect 3 Lambda calls."""
    fake = FakeLambdaClient()
    fake.set_response({"processed_count": 2, "skipped_count": 0, "total_chunks_added": 4})
    service = _make_service(fake)

    docs = [{"filename": f"doc{i}.md", "content": f"content {i}"} for i in range(5)]
    result = asyncio.run(service.build_index(documents=docs, batch_size=2))

    assert len(fake.calls) == 3  # 2 + 2 + 1
    # Totals aggregated across 3 calls (each returning 2 processed, 4 chunks)
    assert result["processed_count"] == 6
    assert result["total_chunks_added"] == 12


def test_build_index_aggregates_results():
    fake = FakeLambdaClient()
    call_count = {"n": 0}
    original_invoke = fake.invoke

    def _invoke(**kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            fake.set_response({"processed_count": 3, "skipped_count": 1, "total_chunks_added": 15})
        else:
            fake.set_response({"processed_count": 2, "skipped_count": 0, "total_chunks_added": 8})
        return original_invoke(**kwargs)

    fake.invoke = _invoke  # type: ignore[method-assign]
    service = _make_service(fake)

    docs = [{"filename": f"doc{i}.md", "content": f"content {i}"} for i in range(6)]
    result = asyncio.run(service.build_index(documents=docs, batch_size=3))

    assert result["processed_count"] == 5
    assert result["skipped_count"] == 1
    assert result["total_chunks_added"] == 23


def test_build_index_wraps_errors():
    fake = FakeLambdaClient()
    fake.set_error("timeout")
    service = _make_service(fake)

    with pytest.raises(SearchServiceError, match="Lambda execution failed"):
        asyncio.run(service.build_index(documents=[{"filename": "a.md", "content": "x"}]))
