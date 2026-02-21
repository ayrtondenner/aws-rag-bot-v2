"""Service configuration types.

This package is intentionally kept small: it holds plain dataclasses that are
loaded from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class S3Config:

    bucket_name: str
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None

    @staticmethod
    def from_env() -> "S3Config":
        """Create an S3Config from environment variables.

        Returns:
            An S3Config populated from env vars.

        Raises:
            ValueError: If S3_BUCKET_NAME is not set.
        """

        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("Missing required environment variable: S3_BUCKET_NAME")

        region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        endpoint_url = os.getenv("S3_ENDPOINT_URL")

        return S3Config(bucket_name=bucket_name, region_name=region_name, endpoint_url=endpoint_url)


@dataclass(frozen=True)
class OpenSearchConfig:
    """Configuration for connecting to an OpenSearch Serverless collection."""

    endpoint: str
    index_name: str
    region: Optional[str] = None
    service_name: str = "aoss"
    timeout_seconds: int = 30

    @staticmethod
    def from_env() -> "OpenSearchConfig":
        """Create an OpenSearchConfig from environment variables.

        Returns:
            An OpenSearchConfig populated from env vars.

        Raises:
            ValueError: If required env vars are missing.
        """

        endpoint = (os.getenv("OPENSEARCH_ENDPOINT") or "").strip()
        if not endpoint:
            raise ValueError("Missing required environment variable: OPENSEARCH_ENDPOINT")

        index_name = (os.getenv("OPENSEARCH_INDEX_NAME") or "").strip()
        if not index_name:
            raise ValueError("Missing required environment variable: OPENSEARCH_INDEX_NAME")

        region = (
            os.getenv("OPENSEARCH_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
        )

        service_name = (os.getenv("OPENSEARCH_SERVICE_NAME") or "").strip() or "aoss"

        timeout_raw = (os.getenv("OPENSEARCH_TIMEOUT_SECONDS") or "").strip()
        timeout_seconds = 30
        if timeout_raw:
            try:
                timeout_seconds = int(timeout_raw)
            except ValueError as exc:
                raise ValueError("OPENSEARCH_TIMEOUT_SECONDS must be an integer") from exc

        return OpenSearchConfig(
            endpoint=endpoint,
            index_name=index_name,
            region=region,
            service_name=service_name,
            timeout_seconds=timeout_seconds,
        )