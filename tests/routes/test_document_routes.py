from __future__ import annotations

from typing import Any

import pytest

from app.services.dependencies import get_document_service
from app.services.document_service import DocumentServiceError


class StubDocumentService:
    def __init__(self):
        self.local_docs_result: list[str] = ["a.md", "b.md"]
        self.local_docs_exc: Exception | None = None

        self.chunk_text_result: list[str] = ["a", "b"]
        self.chunk_text_exc: Exception | None = None

        self.embed_text_result: list[float] = [0.1, 0.2, 0.3]
        self.embed_text_exc: Exception | None = None

        self.seen: dict[str, object] = {}

    def list_local_sagemaker_docs(self) -> dict[str, Any]:
        self.seen["list_local_sagemaker_docs.called"] = True
        if self.local_docs_exc is not None:
            raise self.local_docs_exc
        return {"count": len(self.local_docs_result), "documents": self.local_docs_result}

    def chunk_text(self, *, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
        if self.chunk_text_exc is not None:
            raise self.chunk_text_exc
        return self.chunk_text_result

    def embed_text(self, *, text: str) -> list[float]:
        if self.embed_text_exc is not None:
            raise self.embed_text_exc
        return self.embed_text_result


def test_document_chunks_success(client, fastapi_app):
    stub = StubDocumentService()
    stub.chunk_text_result = ["c1", "c2", "c3"]

    fastapi_app.dependency_overrides[get_document_service] = lambda: stub
    try:
        resp = client.post("/document/chunks?chunk_size=500&chunk_overlap=50", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.json() == {
            "count": 3,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "chunks": ["c1", "c2", "c3"],
        }
    finally:
        fastapi_app.dependency_overrides.clear()


def test_document_local_docs_success(client, fastapi_app):
    stub = StubDocumentService()
    stub.local_docs_result = ["a.md", "b.md"]

    fastapi_app.dependency_overrides[get_document_service] = lambda: stub
    try:
        resp = client.get("/document/local-docs")
        assert resp.status_code == 200
        assert resp.json() == {"count": 2, "documents": ["a.md", "b.md"]}
        assert stub.seen["list_local_sagemaker_docs.called"] is True
    finally:
        fastapi_app.dependency_overrides.clear()


def test_document_chunks_maps_value_error_to_400(client, fastapi_app):
    stub = StubDocumentService()
    stub.chunk_text_exc = ValueError("chunk_overlap must be < chunk_size")

    fastapi_app.dependency_overrides[get_document_service] = lambda: stub
    try:
        resp = client.post("/document/chunks?chunk_size=10&chunk_overlap=10", json={"text": "hello"})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "chunk_overlap must be < chunk_size"
    finally:
        fastapi_app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "query",
    [
        "chunk_size=0&chunk_overlap=0",
        "chunk_size=10&chunk_overlap=-1",
    ],
)
def test_document_chunks_invalid_query_params_return_422(client, query: str):
    resp = client.post(f"/document/chunks?{query}", json={"text": "hello"})
    assert resp.status_code == 422


def test_document_embed_success(client, fastapi_app):
    stub = StubDocumentService()
    stub.embed_text_result = [1.0, 2.0]

    fastapi_app.dependency_overrides[get_document_service] = lambda: stub
    try:
        resp = client.post("/document/embed", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.json() == {"dimensions": 2, "embedding": [1.0, 2.0]}
    finally:
        fastapi_app.dependency_overrides.clear()


def test_document_embed_maps_value_error_to_400(client, fastapi_app):
    stub = StubDocumentService()
    stub.embed_text_exc = ValueError("text must be provided")

    fastapi_app.dependency_overrides[get_document_service] = lambda: stub
    try:
        resp = client.post("/document/embed", json={"text": "  "})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "text must be provided"
    finally:
        fastapi_app.dependency_overrides.clear()


def test_document_embed_maps_service_error_to_502(client, fastapi_app):
    stub = StubDocumentService()
    stub.embed_text_exc = DocumentServiceError("Failed to generate embedding")

    fastapi_app.dependency_overrides[get_document_service] = lambda: stub
    try:
        resp = client.post("/document/embed", json={"text": "hello"})
        assert resp.status_code == 502
        assert resp.json()["detail"] == "Failed to generate embedding"
    finally:
        fastapi_app.dependency_overrides.clear()
