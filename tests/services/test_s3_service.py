import asyncio
from datetime import datetime, timezone

import pytest
from botocore.exceptions import ClientError

from app.services.config import S3Config
from app.services.s3_service import S3Service, S3ServiceError


class FakeS3Body:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self) -> bytes:
        return self._data


class FakeS3Client:
    """A minimal async context manager that emulates the aioboto3 S3 client."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

        self.head_bucket_exc: Exception | None = None

        self.list_objects_response: dict | Exception | None = None

        self.put_object_exc: Exception | None = None

        self.get_object_response: dict | Exception | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def head_bucket(self, **kwargs):
        self.calls.append(("head_bucket", kwargs))
        if self.head_bucket_exc is not None:
            raise self.head_bucket_exc
        return {}

    async def list_objects_v2(self, **kwargs):
        self.calls.append(("list_objects_v2", kwargs))
        if isinstance(self.list_objects_response, Exception):
            raise self.list_objects_response
        return self.list_objects_response or {}

    async def put_object(self, **kwargs):
        self.calls.append(("put_object", kwargs))
        if self.put_object_exc is not None:
            raise self.put_object_exc
        return {"ETag": "fake"}

    async def get_object(self, **kwargs):
        self.calls.append(("get_object", kwargs))
        if isinstance(self.get_object_response, Exception):
            raise self.get_object_response
        return self.get_object_response or {}


def _make_client_error(*, code: str, status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "boom"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation_name="X",
    )


def _service_with_fake_client(fake: FakeS3Client) -> S3Service:
    service = S3Service(S3Config(bucket_name="unit-test-bucket", region_name="us-east-1"))
    service._client = lambda: fake  # type: ignore[method-assign]
    return service

# TODO: implement dependency injection to avoid needing to test S3Service directly
def test_bucket_exists_raises_on_blank_name():
    service = S3Service(S3Config(bucket_name="unit-test-bucket"))
    with pytest.raises(ValueError):
        asyncio.run(service.bucket_exists(bucket_name=""))

    with pytest.raises(ValueError):
        asyncio.run(service.bucket_exists(bucket_name="   "))


def test_bucket_exists_true_when_head_bucket_succeeds():
    fake = FakeS3Client()
    service = _service_with_fake_client(fake)

    assert asyncio.run(service.bucket_exists(bucket_name="my-bucket")) is True

    assert fake.calls == [("head_bucket", {"Bucket": "my-bucket"})]


def test_bucket_exists_false_when_not_found_by_status_code():
    fake = FakeS3Client()
    fake.head_bucket_exc = _make_client_error(code="NotFound", status=404)
    service = _service_with_fake_client(fake)

    assert asyncio.run(service.bucket_exists(bucket_name="missing")) is False


def test_bucket_exists_false_when_not_found_by_error_code():
    fake = FakeS3Client()
    fake.head_bucket_exc = _make_client_error(code="NoSuchBucket", status=400)
    service = _service_with_fake_client(fake)

    assert asyncio.run(service.bucket_exists(bucket_name="missing")) is False


def test_bucket_exists_raises_on_access_denied():
    fake = FakeS3Client()
    fake.head_bucket_exc = _make_client_error(code="AccessDenied", status=403)
    service = _service_with_fake_client(fake)

    with pytest.raises(S3ServiceError, match="Access denied"):
        asyncio.run(service.bucket_exists(bucket_name="secret"))


def test_bucket_exists_wraps_unexpected_client_error():
    fake = FakeS3Client()
    fake.head_bucket_exc = _make_client_error(code="InternalError", status=500)
    service = _service_with_fake_client(fake)

    with pytest.raises(S3ServiceError, match="Failed to check if S3 bucket exists"):
        asyncio.run(service.bucket_exists(bucket_name="oops"))


def test_list_files_returns_file_items():
    fake = FakeS3Client()
    fake.list_objects_response = {
        "Contents": [
            {
                "Key": "docs/a.md",
                "Size": 123,
                "LastModified": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "ETag": '"abc"',
            }
        ]
    }
    service = _service_with_fake_client(fake)

    files = asyncio.run(service.list_files(prefix="docs/", max_keys=10))

    assert [f.key for f in files] == ["docs/a.md"]
    assert files[0].size == 123

    # Ensure Prefix is only passed when provided
    assert fake.calls[0][0] == "list_objects_v2"
    assert fake.calls[0][1]["Bucket"] == "unit-test-bucket"
    assert fake.calls[0][1]["MaxKeys"] == 10
    assert fake.calls[0][1]["Prefix"] == "docs/"


def test_list_files_empty_when_no_contents_key():
    fake = FakeS3Client()
    fake.list_objects_response = {}
    service = _service_with_fake_client(fake)

    assert asyncio.run(service.list_files()) == []


def test_list_files_does_not_pass_prefix_when_none_or_empty():
    fake = FakeS3Client()
    fake.list_objects_response = {"Contents": []}
    service = _service_with_fake_client(fake)

    asyncio.run(service.list_files(prefix=None, max_keys=7))
    kwargs = fake.calls[-1][1]
    assert "Prefix" not in kwargs

    asyncio.run(service.list_files(prefix="", max_keys=7))
    kwargs = fake.calls[-1][1]
    assert "Prefix" not in kwargs


def test_list_files_wraps_errors():
    fake = FakeS3Client()
    fake.list_objects_response = RuntimeError("boom")
    service = _service_with_fake_client(fake)

    with pytest.raises(S3ServiceError, match="Failed to list files"):
        asyncio.run(service.list_files())


def test_list_files_handles_malformed_contents_value():
    fake = FakeS3Client()
    fake.list_objects_response = {"Contents": None}
    service = _service_with_fake_client(fake)

    with pytest.raises(S3ServiceError):
        asyncio.run(service.list_files())


def test_upload_local_file_uploads_bytes_and_guesses_content_type(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")

    fake = FakeS3Client()
    service = _service_with_fake_client(fake)

    key = asyncio.run(service.upload_local_file(path=path, key="docs/note.txt"))
    assert key == "docs/note.txt"

    method, kwargs = fake.calls[-1]
    assert method == "put_object"
    assert kwargs["Bucket"] == "unit-test-bucket"
    assert kwargs["Key"] == "docs/note.txt"
    assert kwargs["Body"] == b"hello"
    assert kwargs.get("ContentType") in {"text/plain", "text/plain; charset=utf-8"}


def test_upload_local_file_does_not_set_content_type_when_unknown_extension(tmp_path):
    path = tmp_path / "blob.unknownext"
    path.write_bytes(b"xyz")

    fake = FakeS3Client()
    service = _service_with_fake_client(fake)

    asyncio.run(service.upload_local_file(path=path, key="blob"))

    method, kwargs = fake.calls[-1]
    assert method == "put_object"
    assert "ContentType" not in kwargs


def test_upload_local_file_respects_explicit_content_type(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")

    fake = FakeS3Client()
    service = _service_with_fake_client(fake)

    asyncio.run(service.upload_local_file(path=path, key="data.bin", content_type="application/octet-stream"))

    method, kwargs = fake.calls[-1]
    assert method == "put_object"
    assert kwargs["ContentType"] == "application/octet-stream"


def test_upload_local_file_wraps_validation_errors(tmp_path):
    path = tmp_path / "ok.txt"
    path.write_text("ok", encoding="utf-8")

    fake = FakeS3Client()
    service = _service_with_fake_client(fake)

    with pytest.raises(S3ServiceError, match="Failed to upload"):
        asyncio.run(service.upload_local_file(path=path, key=""))


def test_upload_local_file_wraps_missing_file(tmp_path):
    missing = tmp_path / "missing.txt"

    fake = FakeS3Client()
    service = _service_with_fake_client(fake)

    with pytest.raises(S3ServiceError, match="Failed to upload"):
        asyncio.run(service.upload_local_file(path=missing, key="k"))


def test_get_file_content_reads_body_bytes():
    fake = FakeS3Client()
    fake.get_object_response = {"Body": FakeS3Body(b"hello")}
    service = _service_with_fake_client(fake)

    data = asyncio.run(service.get_file_content(key="docs/a.md"))
    assert data == "hello"

    assert fake.calls[-1] == (
        "get_object",
        {"Bucket": "unit-test-bucket", "Key": "docs/a.md"},
    )


def test_get_file_content_returns_empty_bytes_when_body_missing():
    fake = FakeS3Client()
    fake.get_object_response = {"Body": None}
    service = _service_with_fake_client(fake)

    assert asyncio.run(service.get_file_content(key="k")) == ""


def test_get_file_content_wraps_validation_errors():
    fake = FakeS3Client()
    service = _service_with_fake_client(fake)

    with pytest.raises(S3ServiceError, match="Failed to fetch"):
        asyncio.run(service.get_file_content(key=""))


def test_get_file_content_wraps_client_errors():
    fake = FakeS3Client()
    fake.get_object_response = RuntimeError("boom")
    service = _service_with_fake_client(fake)

    with pytest.raises(S3ServiceError, match="Failed to fetch"):
        asyncio.run(service.get_file_content(key="k"))
