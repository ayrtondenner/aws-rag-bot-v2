from __future__ import annotations

from typing import Any, Literal

from app.models.opensearch import (
    BulkIndexResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    IndexStatsResponse,
    SearchHit,
    SearchResponse,
)
from app.services.dependencies import get_opensearch_service
from app.services.opensearch_service import OpenSearchServiceError


# ---------------------------------------------------------------------------
# Stub OpenSearchService
# ---------------------------------------------------------------------------


class StubOpenSearchService:
    """Test double for ``OpenSearchService`` — configurable canned responses."""

    def __init__(self) -> None:
        # document_exists
        self.document_exists_result: bool = False
        self.document_exists_exc: Exception | None = None

        # list_indexed_documents
        self.list_indexed_docs_result: list[str] = []
        self.list_indexed_docs_exc: Exception | None = None

        # index_document
        self.index_document_result: IndexDocumentResponse = IndexDocumentResponse(
            filename="stub.md", chunk_count=2, doc_ids=["id-1", "id-2"], skipped=False,
        )
        self.index_document_exc: Exception | None = None

        # bulk_index_documents
        self.bulk_index_result: BulkIndexResponse = BulkIndexResponse(
            total_chunks=0, indexed_count=0, skipped_count=0, results=[],
        )
        self.bulk_index_exc: Exception | None = None

        # search
        self.search_result: SearchResponse = SearchResponse(
            query="", search_type="hybrid", total_hits=0, hits=[],
        )
        self.search_exc: Exception | None = None

        # delete_document
        self.delete_document_result: bool = True
        self.delete_document_exc: Exception | None = None

        # delete_documents_by_filename
        self.delete_by_filename_result: int = 0
        self.delete_by_filename_exc: Exception | None = None

        # get_index_stats
        self.index_stats_result: IndexStatsResponse = IndexStatsResponse(
            index_name="test-index", doc_count=0, status="available",
        )
        self.index_stats_exc: Exception | None = None

        self.seen: dict[str, Any] = {}

    async def document_exists(self, *, filename: str) -> bool:
        self.seen["document_exists.filename"] = filename
        if self.document_exists_exc is not None:
            raise self.document_exists_exc
        return self.document_exists_result

    async def list_indexed_documents(self) -> list[str]:
        if self.list_indexed_docs_exc is not None:
            raise self.list_indexed_docs_exc
        return self.list_indexed_docs_result

    async def index_document(
        self, *, filename: str, content: str, chunk_size: int = 500, chunk_overlap: int = 50,
    ) -> IndexDocumentResponse:
        self.seen["index_document.filename"] = filename
        self.seen["index_document.content"] = content
        if self.index_document_exc is not None:
            raise self.index_document_exc
        return self.index_document_result

    async def bulk_index_documents(
        self, *, documents: list[IndexDocumentRequest], chunk_size: int = 500, chunk_overlap: int = 50,
    ) -> BulkIndexResponse:
        self.seen["bulk_index.documents"] = documents
        if self.bulk_index_exc is not None:
            raise self.bulk_index_exc
        return self.bulk_index_result

    async def search(
        self, *, query: str, size: int = 10, search_type: Literal["hybrid", "text", "vector"] = "hybrid",
    ) -> SearchResponse:
        self.seen["search.query"] = query
        self.seen["search.size"] = size
        self.seen["search.search_type"] = search_type
        if self.search_exc is not None:
            raise self.search_exc
        return self.search_result

    async def delete_document(self, *, doc_id: str) -> bool:
        self.seen["delete_document.doc_id"] = doc_id
        if self.delete_document_exc is not None:
            raise self.delete_document_exc
        return self.delete_document_result

    async def delete_documents_by_filename(self, *, filename: str) -> int:
        self.seen["delete_by_filename.filename"] = filename
        if self.delete_by_filename_exc is not None:
            raise self.delete_by_filename_exc
        return self.delete_by_filename_result

    async def get_index_stats(self) -> IndexStatsResponse:
        if self.index_stats_exc is not None:
            raise self.index_stats_exc
        return self.index_stats_result


# ===========================================================================
# Index single document
# ===========================================================================


def test_index_document_success(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.index_document_result = IndexDocumentResponse(
        filename="doc.md", chunk_count=3, doc_ids=["a", "b", "c"], skipped=False,
    )
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.post("/opensearch/index", json={"filename": "doc.md", "content": "my text"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "doc.md"
        assert data["chunk_count"] == 3
        assert data["skipped"] is False
        assert stub.seen["index_document.filename"] == "doc.md"
    finally:
        fastapi_app.dependency_overrides.clear()


def test_index_document_skipped(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.index_document_result = IndexDocumentResponse(
        filename="doc.md", chunk_count=0, doc_ids=[], skipped=True,
    )
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.post("/opensearch/index", json={"filename": "doc.md", "content": "text"})
        assert resp.status_code == 200
        assert resp.json()["skipped"] is True
    finally:
        fastapi_app.dependency_overrides.clear()


def test_index_document_value_error_returns_400(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.index_document_exc = ValueError("content must be provided")
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.post("/opensearch/index", json={"filename": "x.md", "content": "t"})
        assert resp.status_code == 400
        assert "content must be provided" in resp.json()["detail"]
    finally:
        fastapi_app.dependency_overrides.clear()


def test_index_document_service_error_returns_502(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.index_document_exc = OpenSearchServiceError("bulk failed")
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.post("/opensearch/index", json={"filename": "x.md", "content": "t"})
        assert resp.status_code == 502
        assert resp.json()["detail"] == "bulk failed"
    finally:
        fastapi_app.dependency_overrides.clear()


# ===========================================================================
# Bulk-index
# ===========================================================================


def test_bulk_index_success(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.bulk_index_result = BulkIndexResponse(
        total_chunks=5, indexed_count=2, skipped_count=1,
        results=[
            IndexDocumentResponse(filename="a.md", chunk_count=3, doc_ids=["1", "2", "3"], skipped=False),
            IndexDocumentResponse(filename="b.md", chunk_count=2, doc_ids=["4", "5"], skipped=False),
            IndexDocumentResponse(filename="c.md", chunk_count=0, doc_ids=[], skipped=True),
        ],
    )
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.post("/opensearch/bulk-index", json={
            "documents": [
                {"filename": "a.md", "content": "aaa"},
                {"filename": "b.md", "content": "bbb"},
                {"filename": "c.md", "content": "ccc"},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["indexed_count"] == 2
        assert data["skipped_count"] == 1
        assert data["total_chunks"] == 5
    finally:
        fastapi_app.dependency_overrides.clear()


# ===========================================================================
# Search
# ===========================================================================


def test_search_hybrid_success(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.search_result = SearchResponse(
        query="sagemaker", search_type="hybrid", total_hits=1,
        hits=[SearchHit(doc_id="h1", score=0.9, filename="doc.md", content="found")],
    )
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.post("/opensearch/search", json={"query": "sagemaker"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_hits"] == 1
        assert data["hits"][0]["doc_id"] == "h1"
        assert stub.seen["search.query"] == "sagemaker"
        assert stub.seen["search.search_type"] == "hybrid"
    finally:
        fastapi_app.dependency_overrides.clear()


def test_search_text_only(client, fastapi_app):
    stub = StubOpenSearchService()
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.post("/opensearch/search", json={"query": "test", "search_type": "text"})
        assert resp.status_code == 200
        assert stub.seen["search.search_type"] == "text"
    finally:
        fastapi_app.dependency_overrides.clear()


def test_search_empty_query_returns_422(client, fastapi_app):
    stub = StubOpenSearchService()
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.post("/opensearch/search", json={"query": ""})
        assert resp.status_code == 422
    finally:
        fastapi_app.dependency_overrides.clear()


def test_search_service_error_returns_502(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.search_exc = OpenSearchServiceError("search failed")
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.post("/opensearch/search", json={"query": "test"})
        assert resp.status_code == 502
        assert resp.json()["detail"] == "search failed"
    finally:
        fastapi_app.dependency_overrides.clear()


# ===========================================================================
# Index stats
# ===========================================================================


def test_index_stats_success(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.index_stats_result = IndexStatsResponse(
        index_name="sagemaker-docs", doc_count=42, status="available",
    )
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.get("/opensearch/index/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_count"] == 42
        assert data["index_name"] == "sagemaker-docs"
    finally:
        fastapi_app.dependency_overrides.clear()


def test_index_stats_service_error_returns_502(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.index_stats_exc = OpenSearchServiceError("down")
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.get("/opensearch/index/stats")
        assert resp.status_code == 502
    finally:
        fastapi_app.dependency_overrides.clear()


# ===========================================================================
# List indexed documents
# ===========================================================================


def test_list_indexed_documents_success(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.list_indexed_docs_result = ["a.md", "b.md"]
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.get("/opensearch/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["filenames"] == ["a.md", "b.md"]
    finally:
        fastapi_app.dependency_overrides.clear()


# ===========================================================================
# Document exists
# ===========================================================================


def test_document_exists_true(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.document_exists_result = True
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.get("/opensearch/document/exists?filename=doc.md")
        assert resp.status_code == 200
        assert resp.json() == {"filename": "doc.md", "exists": True}
    finally:
        fastapi_app.dependency_overrides.clear()


def test_document_exists_false(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.document_exists_result = False
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.get("/opensearch/document/exists?filename=missing.md")
        assert resp.status_code == 200
        assert resp.json()["exists"] is False
    finally:
        fastapi_app.dependency_overrides.clear()


def test_document_exists_missing_param_returns_422(client, fastapi_app):
    stub = StubOpenSearchService()
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.get("/opensearch/document/exists")
        assert resp.status_code == 422
    finally:
        fastapi_app.dependency_overrides.clear()


# ===========================================================================
# Delete single document
# ===========================================================================


def test_delete_document_success(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.delete_document_result = True
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.delete("/opensearch/document/abc123")
        assert resp.status_code == 200
        assert resp.json() == {"deleted": True}
        assert stub.seen["delete_document.doc_id"] == "abc123"
    finally:
        fastapi_app.dependency_overrides.clear()


def test_delete_document_not_found_returns_404(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.delete_document_result = False
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.delete("/opensearch/document/missing")
        assert resp.status_code == 404
    finally:
        fastapi_app.dependency_overrides.clear()


# ===========================================================================
# Delete by filename
# ===========================================================================


def test_delete_by_filename_success(client, fastapi_app):
    stub = StubOpenSearchService()
    stub.delete_by_filename_result = 5
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.delete("/opensearch/documents?filename=doc.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "doc.md"
        assert data["deleted_count"] == 5
    finally:
        fastapi_app.dependency_overrides.clear()


def test_delete_by_filename_missing_param_returns_422(client, fastapi_app):
    stub = StubOpenSearchService()
    fastapi_app.dependency_overrides[get_opensearch_service] = lambda: stub
    try:
        resp = client.delete("/opensearch/documents")
        assert resp.status_code == 422
    finally:
        fastapi_app.dependency_overrides.clear()
