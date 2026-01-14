from __future__ import annotations

import os
from typing import Any

from botocore.exceptions import ClientError

from app.services.config import S3Config
from app.services.s3_service import S3Service, S3ServiceError


class S3SetupServiceError(RuntimeError):
    pass


class S3SetupService:
    def __init__(self, *, s3: S3Service) -> None:
        self._s3 = s3

    @staticmethod
    def from_env() -> "S3SetupService":
        return S3SetupService(s3=S3Service(S3Config.from_env()))

    async def setup_bucket(self) -> str:
        """Ensure the configured S3 bucket exists, creating it if missing.

        Uses the env var `S3_BUCKET_NAME` (via S3Config.from_env()).

        Returns:
            The bucket name.
        """

        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            raise S3SetupServiceError("Missing required environment variable: S3_BUCKET_NAME")

        exists = await self._s3.bucket_exists(bucket_name=bucket_name)
        if exists:
            return bucket_name

        await self._create_bucket(bucket_name=bucket_name)
        return bucket_name

    async def _create_bucket(self, *, bucket_name: str) -> None:
        # CreateBucket is region-specific. For us-east-1, you must omit CreateBucketConfiguration.
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

        try:
            s3_client: Any = self._s3._client()
            async with s3_client as s3:
                if region and region != "us-east-1":
                    await s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": region},
                    )
                else:
                    await s3.create_bucket(Bucket=bucket_name)
        except ClientError as exc:
            code = (exc.response.get("Error") or {}).get("Code")
            # If a concurrent process created it, treat as success.
            if code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                return
            raise S3SetupServiceError(f"Failed to create S3 bucket: {bucket_name} (code={code})") from exc
        except S3ServiceError as exc:
            raise S3SetupServiceError(f"Failed to create S3 bucket: {bucket_name}") from exc
        except Exception as exc:
            raise S3SetupServiceError(f"Failed to create S3 bucket: {bucket_name}") from exc
