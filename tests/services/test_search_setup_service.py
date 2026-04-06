from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest

from app.models.search import (
    BulkIndexResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
)
from app.services.config import SearchConfig
from app.services.document_service import DocumentService
from app.services.search_service import SearchServiceError
from app.services.setup.search_setup_service import SearchSetupService


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


class StubSearchService:
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

_DEFAULT_CONFIG = SearchConfig(
    lambda_function_name="test-lambda",
    index_bucket="test-bucket",
    index_prefix="test-index/",
    region="us-east-1",
)


def _make_setup_service(
    *,
    doc_stub: StubDocumentService,
    search_stub: StubSearchService,
    lambda_exists: bool = True,
    artifacts_exist: bool = True,
) -> SearchSetupService:
    service = SearchSetupService(
        search=search_stub,  # type: ignore[arg-type]
        document_service=doc_stub,
        config=_DEFAULT_CONFIG,
    )
    # Override existence checks to avoid real AWS calls
    service._lambda_exists = lambda: lambda_exists  # type: ignore[method-assign]
    service._artifacts_exist = lambda: artifacts_exist  # type: ignore[method-assign]
    # Override build/deploy to avoid real AWS calls
    service._build_and_upload_artifacts = lambda: None  # type: ignore[method-assign]
    service._create_or_update_lambda = lambda: None  # type: ignore[method-assign]
    return service


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_setup_when_both_exist_indexes_local_docs():
    """Happy path: Lambda and artifacts exist, just bulk-index local docs."""
    doc_stub = StubDocumentService(
        local_docs={"count": 2, "documents": ["a.md", "b.md"]},
        file_contents={"a.md": "content a", "b.md": "content b"},
    )
    search_stub = StubSearchService(
        bulk_response=BulkIndexResponse(
            total_chunks=4, indexed_count=2, skipped_count=0,
            results=[
                IndexDocumentResponse(filename="a.md", chunk_count=2, doc_ids=["1", "2"], skipped=False),
                IndexDocumentResponse(filename="b.md", chunk_count=2, doc_ids=["3", "4"], skipped=False),
            ],
        ),
    )
    service = _make_setup_service(
        doc_stub=doc_stub, search_stub=search_stub,
        lambda_exists=True, artifacts_exist=True,
    )

    result = asyncio.run(service.setup_index())

    assert result.indexed_count == 2
    assert result.total_chunks == 4
    assert len(search_stub.calls) == 1


def test_setup_no_local_docs_returns_empty():
    doc_stub = StubDocumentService(local_docs={"count": 0, "documents": []})
    search_stub = StubSearchService()
    service = _make_setup_service(
        doc_stub=doc_stub, search_stub=search_stub,
        lambda_exists=True, artifacts_exist=True,
    )

    result = asyncio.run(service.setup_index())

    assert result.indexed_count == 0
    assert result.total_chunks == 0
    assert search_stub.calls == []


def test_setup_skips_unreadable_files():
    doc_stub = StubDocumentService(
        local_docs={"count": 3, "documents": ["good.md", "bad.md", "ok.md"]},
        file_contents={"good.md": "good content", "ok.md": "ok content"},
        unreadable_files={"bad.md"},
    )
    search_stub = StubSearchService(
        bulk_response=BulkIndexResponse(
            total_chunks=4, indexed_count=2, skipped_count=0,
            results=[
                IndexDocumentResponse(filename="good.md", chunk_count=2, doc_ids=["1", "2"], skipped=False),
                IndexDocumentResponse(filename="ok.md", chunk_count=2, doc_ids=["3", "4"], skipped=False),
            ],
        ),
    )
    service = _make_setup_service(
        doc_stub=doc_stub, search_stub=search_stub,
        lambda_exists=True, artifacts_exist=True,
    )

    result = asyncio.run(service.setup_index())

    assert result.indexed_count == 2
    filenames_sent = [d.filename for d in search_stub.calls[0]["documents"]]
    assert "bad.md" not in filenames_sent


def test_setup_propagates_search_errors():
    doc_stub = StubDocumentService(
        local_docs={"count": 1, "documents": ["a.md"]},
        file_contents={"a.md": "content"},
    )
    search_stub = StubSearchService(error=SearchServiceError("connection refused"))
    service = _make_setup_service(
        doc_stub=doc_stub, search_stub=search_stub,
        lambda_exists=True, artifacts_exist=True,
    )

    with pytest.raises(SearchServiceError, match="connection refused"):
        asyncio.run(service.setup_index())


def test_setup_missing_artifacts_triggers_build():
    """When artifacts don't exist, build is triggered."""
    build_called = {"value": False}
    doc_stub = StubDocumentService(local_docs={"count": 0, "documents": []})
    search_stub = StubSearchService()
    service = _make_setup_service(
        doc_stub=doc_stub, search_stub=search_stub,
        lambda_exists=True, artifacts_exist=False,
    )
    def _tracking_build() -> None:
        build_called["value"] = True

    service._build_and_upload_artifacts = _tracking_build  # type: ignore[method-assign]

    asyncio.run(service.setup_index())
    assert build_called["value"] is True


def test_setup_missing_lambda_triggers_deploy():
    """When Lambda doesn't exist, deployment is triggered."""
    deploy_called = {"value": False}
    doc_stub = StubDocumentService(local_docs={"count": 0, "documents": []})
    search_stub = StubSearchService()
    service = _make_setup_service(
        doc_stub=doc_stub, search_stub=search_stub,
        lambda_exists=False, artifacts_exist=True,
    )

    def _tracking_deploy() -> None:
        deploy_called["value"] = True

    service._create_or_update_lambda = _tracking_deploy  # type: ignore[method-assign]

    asyncio.run(service.setup_index())
    assert deploy_called["value"] is True


def test_setup_with_dedup_results():
    doc_stub = StubDocumentService(
        local_docs={"count": 2, "documents": ["new.md", "old.md"]},
        file_contents={"new.md": "new content", "old.md": "old content"},
    )
    search_stub = StubSearchService(
        bulk_response=BulkIndexResponse(
            total_chunks=2, indexed_count=1, skipped_count=1,
            results=[
                IndexDocumentResponse(filename="new.md", chunk_count=2, doc_ids=["1", "2"], skipped=False),
                IndexDocumentResponse(filename="old.md", chunk_count=0, doc_ids=[], skipped=True),
            ],
        ),
    )
    service = _make_setup_service(
        doc_stub=doc_stub, search_stub=search_stub,
        lambda_exists=True, artifacts_exist=True,
    )

    result = asyncio.run(service.setup_index())

    assert result.indexed_count == 1
    assert result.skipped_count == 1
    assert result.total_chunks == 2
