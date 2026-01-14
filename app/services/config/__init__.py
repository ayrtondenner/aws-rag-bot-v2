"""Service configuration types.

This package is intentionally kept small: it holds plain dataclasses that are
loaded from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Optional


@dataclass(frozen=True)
class S3Config:

    bucket_name: str
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None

    @staticmethod
    def from_env() -> "S3Config":
        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("Missing required environment variable: S3_BUCKET_NAME")

        region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        endpoint_url = os.getenv("S3_ENDPOINT_URL")

        return S3Config(bucket_name=bucket_name, region_name=region_name, endpoint_url=endpoint_url)