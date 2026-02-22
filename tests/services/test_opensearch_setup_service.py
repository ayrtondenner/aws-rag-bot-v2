from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest

from app.models.opensearch import (
    BulkIndexResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
)
from app.services.document_service import DocumentService
from app.services.opensearch_service import OpenSearchServiceError
from app.services.setup.opensearch_setup_service import OpenSearchSetupService


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubDocumentService(DocumentService):
    """DocumentService stub that returns canned local-doc data."""

    def __init__(
        self,
        *,
        local_docs: Optional[dict[str, object]] = None,
        file_contents: Optional[dict[str, str]] = None,
        unreadable_files: Optional[set[str]] = None,
    ) -> None:
        self._embedding_model_id = ""
        self._embedding_dim = 1024
        self._local_docs = local_docs or {"count": 0, "documents": []}
        self._file_contents = file_contents or {}
        self._unreadable_files = unreadable_files or set()

    def list_local_sagemaker_docs(self) -> dict[str, object]:
        return dict(self._local_docs)

    def get_local_sagemaker_doc_content(self, *, filename: str, encoding: str = "utf-8") -> str:
        if filename in self._unreadable_files:
            raise FileNotFoundError(filename)
        if filename in self._file_contents:
            return self._file_contents[filename]
        raise FileNotFoundError(filename)

    def chunk_text(self, *, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
        return ["chunk-0", "chunk-1"] if text.strip() else []


class StubOpenSearchService:
    """Captures bulk_index_documents calls and returns configurable results."""

    def __init__(
        self,
        *,
        bulk_response: Optional[BulkIndexResponse] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._bulk_response = bulk_response or BulkIndexResponse(
            total_chunks=0, indexed_count=0, skipped_count=0, results=[],
        )
        self._error = error

    async def bulk_index_documents(
        self,
        *,
        documents: list[IndexDocumentRequest],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_concurrency: int = 5,
    ) -> BulkIndexResponse:
        self.calls.append({
            "documents": documents,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_concurrency": max_concurrency,
        })
        if self._error:
            raise self._error
        return self._bulk_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_setup_service(
    *,
    doc_stub: StubDocumentService,
    os_stub: StubOpenSearchService,
) -> OpenSearchSetupService:
    return OpenSearchSetupService(
        opensearch=os_stub,  # type: ignore[arg-type]
        document_service=doc_stub,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_setup_index_indexes_all_local_docs():
    """Happy path: local docs are read and passed to bulk_index_documents."""
    doc_stub = StubDocumentService(
        local_docs={"count": 2, "documents": ["a.md", "b.md"]},
        file_contents={"a.md": "content a", "b.md": "content b"},
    )
    os_stub = StubOpenSearchService(
        bulk_response=BulkIndexResponse(
            total_chunks=4,
            indexed_count=2,
            skipped_count=0,
            results=[
                IndexDocumentResponse(filename="a.md", chunk_count=2, doc_ids=["1", "2"], skipped=False),
                IndexDocumentResponse(filename="b.md", chunk_count=2, doc_ids=["3", "4"], skipped=False),
            ],
        ),
    )
    service = _make_setup_service(doc_stub=doc_stub, os_stub=os_stub)

    result = asyncio.run(service.setup_index())

    assert result.indexed_count == 2
    assert result.total_chunks == 4
    assert len(os_stub.calls) == 1
    assert len(os_stub.calls[0]["documents"]) == 2


def test_setup_index_returns_empty_when_no_local_docs():
    """No local docs -> returns zero-count BulkIndexResponse without calling OpenSearch."""
    doc_stub = StubDocumentService(local_docs={"count": 0, "documents": []})
    os_stub = StubOpenSearchService()
    service = _make_setup_service(doc_stub=doc_stub, os_stub=os_stub)

    result = asyncio.run(service.setup_index())

    assert result.indexed_count == 0
    assert result.skipped_count == 0
    assert result.total_chunks == 0
    assert os_stub.calls == []


def test_setup_index_skips_unreadable_files():
    """Unreadable files are skipped; remaining files are still indexed."""
    doc_stub = StubDocumentService(
        local_docs={"count": 3, "documents": ["good.md", "bad.md", "ok.md"]},
        file_contents={"good.md": "good content", "ok.md": "ok content"},
        unreadable_files={"bad.md"},
    )
    os_stub = StubOpenSearchService(
        bulk_response=BulkIndexResponse(
            total_chunks=4,
            indexed_count=2,
            skipped_count=0,
            results=[
                IndexDocumentResponse(filename="good.md", chunk_count=2, doc_ids=["1", "2"], skipped=False),
                IndexDocumentResponse(filename="ok.md", chunk_count=2, doc_ids=["3", "4"], skipped=False),
            ],
        ),
    )
    service = _make_setup_service(doc_stub=doc_stub, os_stub=os_stub)

    result = asyncio.run(service.setup_index())

    assert result.indexed_count == 2
    assert len(os_stub.calls[0]["documents"]) == 2
    filenames_sent = [d.filename for d in os_stub.calls[0]["documents"]]
    assert "bad.md" not in filenames_sent


def test_setup_index_all_files_unreadable_returns_empty():
    """If every file is unreadable, returns zero-count response."""
    doc_stub = StubDocumentService(
        local_docs={"count": 2, "documents": ["bad1.md", "bad2.md"]},
        file_contents={},
        unreadable_files={"bad1.md", "bad2.md"},
    )
    os_stub = StubOpenSearchService()
    service = _make_setup_service(doc_stub=doc_stub, os_stub=os_stub)

    result = asyncio.run(service.setup_index())

    assert result.indexed_count == 0
    assert os_stub.calls == []


def test_setup_index_propagates_opensearch_errors():
    """If OpenSearchService raises, the error propagates (caller in main.py catches it)."""
    doc_stub = StubDocumentService(
        local_docs={"count": 1, "documents": ["a.md"]},
        file_contents={"a.md": "content"},
    )
    os_stub = StubOpenSearchService(error=OpenSearchServiceError("connection refused"))
    service = _make_setup_service(doc_stub=doc_stub, os_stub=os_stub)

    with pytest.raises(OpenSearchServiceError, match="connection refused"):
        asyncio.run(service.setup_index())


def test_setup_index_with_dedup_results():
    """Verify the service correctly returns results when some docs are skipped (dedup)."""
    doc_stub = StubDocumentService(
        local_docs={"count": 2, "documents": ["new.md", "old.md"]},
        file_contents={"new.md": "new content", "old.md": "old content"},
    )
    os_stub = StubOpenSearchService(
        bulk_response=BulkIndexResponse(
            total_chunks=2,
            indexed_count=1,
            skipped_count=1,
            results=[
                IndexDocumentResponse(filename="new.md", chunk_count=2, doc_ids=["1", "2"], skipped=False),
                IndexDocumentResponse(filename="old.md", chunk_count=0, doc_ids=[], skipped=True),
            ],
        ),
    )
    service = _make_setup_service(doc_stub=doc_stub, os_stub=os_stub)

    result = asyncio.run(service.setup_index())

    assert result.indexed_count == 1
    assert result.skipped_count == 1
    assert result.total_chunks == 2
