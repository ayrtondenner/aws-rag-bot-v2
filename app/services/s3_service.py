from __future__ import annotations

import logging
import mimetypes
from http import HTTPStatus
from pathlib import Path
from typing import Any, Optional

import aioboto3
from botocore.exceptions import ClientError

from app.models.s3 import FileItem
from app.services.config import S3Config

logger = logging.getLogger(__name__)


class S3ServiceError(RuntimeError):
    pass


class S3Service:
    def __init__(self, config: S3Config) -> None:
        self._config = config
        self._session = aioboto3.Session()

    def _client(self) -> Any:
        return self._session.client(
            "s3",
            region_name=self._config.region_name,
            endpoint_url=self._config.endpoint_url,
        )

    async def bucket_exists(self, *, bucket_name: str) -> bool:
        """Return True if the bucket exists (and is accessible), otherwise False.

        Notes:
        - If the bucket exists but is not accessible, AWS commonly returns 403.
          In that case we raise to avoid incorrectly attempting creation.
        """

        if not bucket_name or not bucket_name.strip():
            raise ValueError("bucket_name must be provided")

        try:
            s3_client: Any = self._client()
            async with s3_client as s3:
                await s3.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as exc:
            code = (exc.response.get("Error") or {}).get("Code")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            status_enum = HTTPStatus(status) if isinstance(status, int) else None
            # Common "does not exist" variants:
            if status_enum == HTTPStatus.NOT_FOUND or code in {"NoSuchBucket", "NotFound"}:
                return False
            if status_enum == HTTPStatus.FORBIDDEN or code in {"AccessDenied"}:
                raise S3ServiceError(f"Access denied checking S3 bucket: {bucket_name}") from exc
            logger.exception("S3 bucket_exists failed")
            raise S3ServiceError(f"Failed to check if S3 bucket exists: {bucket_name}") from exc
        except Exception as exc:
            logger.exception("S3 bucket_exists failed")
            raise S3ServiceError(f"Failed to check if S3 bucket exists: {bucket_name}") from exc

    async def list_files(self, *, prefix: Optional[str] = None, max_keys: int = 1000) -> list[FileItem]:
        try:
            kwargs: dict[str, Any] = {"Bucket": self._config.bucket_name, "MaxKeys": max_keys}
            if prefix:
                kwargs["Prefix"] = prefix

            s3_client: Any = self._client()
            async with s3_client as s3:
                response = await s3.list_objects_v2(**kwargs)

            objects = response.get("Contents", [])
            return [FileItem.from_s3_object(o) for o in objects]
        except Exception as exc:
            logger.exception("S3 list_files failed")
            raise S3ServiceError("Failed to list files from S3") from exc

    async def upload_local_file(self, *, path: Path, key: str, content_type: Optional[str] = None) -> str:
        """Upload a local file to S3.

        Args:
            path: Local file path.
            key: Destination S3 object key.
            content_type: Optional content type override.

        Returns:
            The uploaded object key.
        """

        try:
            if not key:
                raise ValueError("'key' must be provided")
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(str(path))

            body = path.read_bytes()
            effective_content_type = content_type
            if effective_content_type is None:
                guessed, _ = mimetypes.guess_type(str(path))
                effective_content_type = guessed

            extra_args: dict[str, Any] = {}
            if effective_content_type:
                extra_args["ContentType"] = effective_content_type

            s3_client: Any = self._client()
            async with s3_client as s3:
                await s3.put_object(
                    Bucket=self._config.bucket_name,
                    Key=key,
                    Body=body,
                    **extra_args,
                )

            return key
        except Exception as exc:
            logger.exception("S3 upload_path failed")
            raise S3ServiceError(f"Failed to upload local file to S3 (key={key})") from exc

    async def get_file_content(self, *, key: str) -> bytes:
        """Return the raw bytes for an object in the configured S3 bucket."""

        try:
            if not key:
                raise ValueError("'key' must be provided")

            s3_client: Any = self._client()
            async with s3_client as s3:
                resp = await s3.get_object(Bucket=self._config.bucket_name, Key=key)
                body = resp.get("Body")
                if body is None:
                    return b""
                return await body.read()
        except Exception as exc:
            logger.exception("S3 get_file_content failed")
            raise S3ServiceError(f"Failed to fetch file content from S3 (key={key})") from exc
