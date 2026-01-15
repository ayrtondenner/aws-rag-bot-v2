from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.models.s3 import FileItem
from app.services.dependencies import get_s3_service
from app.services.s3_service import S3ServiceError


class StubS3Service:
    def __init__(self):
        self.bucket_exists_result: bool = True
        self.bucket_exists_exc: Exception | None = None

        self.list_files_result: list[FileItem] = [FileItem(key="a.txt")]
        self.list_files_exc: Exception | None = None

        self.get_file_content_result: str = "data"
        self.get_file_content_exc: Exception | None = None

        self.seen: dict[str, object] = {}

    async def bucket_exists(self, *, bucket_name: str) -> bool:
        self.seen["bucket_exists.bucket_name"] = bucket_name
        if self.bucket_exists_exc is not None:
            raise self.bucket_exists_exc
        return self.bucket_exists_result

    async def list_files(self, *, prefix=None, max_keys: int = 1000):
        self.seen["list_files.prefix"] = prefix
        self.seen["list_files.max_keys"] = max_keys
        if self.list_files_exc is not None:
            raise self.list_files_exc
        return self.list_files_result

    async def get_file_content(self, *, key: str) -> str:
        self.seen["get_file_content.key"] = key
        if self.get_file_content_exc is not None:
            raise self.get_file_content_exc
        return self.get_file_content_result


def test_s3_bucket_exists_success(client, fastapi_app, monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "unit-test-bucket")

    stub = StubS3Service()
    stub.bucket_exists_result = True

    fastapi_app.dependency_overrides[get_s3_service] = lambda: stub
    try:
        resp = client.get("/s3/bucket/exists")
        assert resp.status_code == 200
        assert resp.json() == {"bucket_name": "unit-test-bucket", "exists": True}
        assert stub.seen["bucket_exists.bucket_name"] == "unit-test-bucket"
    finally:
        fastapi_app.dependency_overrides.clear()


def test_s3_bucket_exists_maps_service_error_to_502(client, fastapi_app, monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "unit-test-bucket")

    stub = StubS3Service()
    stub.bucket_exists_exc = S3ServiceError("boom")

    fastapi_app.dependency_overrides[get_s3_service] = lambda: stub
    try:
        resp = client.get("/s3/bucket/exists")
        assert resp.status_code == 502
        assert resp.json() == {"detail": "boom"}
    finally:
        fastapi_app.dependency_overrides.clear()


def test_s3_bucket_exists_missing_env_var_returns_500(client, monkeypatch):
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
    # The dependency provider raises before the route handler runs.
    # Use raise_server_exceptions=False to assert on the HTTP 500 response.
    resp = TestClient(client.app, raise_server_exceptions=False).get("/s3/bucket/exists")
    assert resp.status_code == 500


def test_s3_bucket_files_count_success(client, fastapi_app, monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "unit-test-bucket")

    stub = StubS3Service()
    stub.list_files_result = [FileItem(key="a"), FileItem(key="b")]

    fastapi_app.dependency_overrides[get_s3_service] = lambda: stub
    try:
        resp = client.get("/s3/bucket/files/count?prefix=docs/")
        assert resp.status_code == 200
        assert resp.json() == {"bucket_name": "unit-test-bucket", "prefix": "docs/", "count": 2}
        assert stub.seen["list_files.prefix"] == "docs/"
    finally:
        fastapi_app.dependency_overrides.clear()


def test_s3_list_files_success(client, fastapi_app):
    stub = StubS3Service()
    stub.list_files_result = [FileItem(key="a.txt"), FileItem(key="b.txt")]

    fastapi_app.dependency_overrides[get_s3_service] = lambda: stub
    try:
        resp = client.get("/s3/files?prefix=docs/")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2
        assert [f["key"] for f in resp.json()["files"]] == ["a.txt", "b.txt"]
        assert stub.seen["list_files.prefix"] == "docs/"
    finally:
        fastapi_app.dependency_overrides.clear()


def test_s3_get_file_content_returns_json(client, fastapi_app):
    stub = StubS3Service()
    stub.get_file_content_result = "hello"

    fastapi_app.dependency_overrides[get_s3_service] = lambda: stub
    try:
        resp = client.get("/s3/file/content?file_name=docs/a.txt")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json() == {"filename": "docs/a.txt", "content": "hello"}
    finally:
        fastapi_app.dependency_overrides.clear()


def test_s3_get_file_content_maps_filenotfound_to_404(client, fastapi_app):
    stub = StubS3Service()
    stub.get_file_content_exc = FileNotFoundError("missing")

    fastapi_app.dependency_overrides[get_s3_service] = lambda: stub
    try:
        resp = client.get("/s3/file/content?file_name=missing")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "File not found"}
    finally:
        fastapi_app.dependency_overrides.clear()


def test_s3_get_file_content_maps_unexpected_error_to_500(client, fastapi_app):
    stub = StubS3Service()
    stub.get_file_content_exc = RuntimeError("boom")

    fastapi_app.dependency_overrides[get_s3_service] = lambda: stub
    try:
        resp = client.get("/s3/file/content?file_name=docs/a.txt")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Failed to fetch file content"}
    finally:
        fastapi_app.dependency_overrides.clear()


def test_s3_get_file_content_query_validation_422(client):
    resp = client.get("/s3/file/content")
    assert resp.status_code == 422

    resp = client.get("/s3/file/content?file_name=")
    assert resp.status_code == 422
