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
class SearchConfig:
    """Configuration for the FAISS + BM25 + S3 + Lambda search backend."""

    lambda_function_name: str
    index_bucket: str
    index_prefix: str = "search-index/"
    region: Optional[str] = None
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    lambda_timeout_seconds: int = 30

    @staticmethod
    def from_env() -> "SearchConfig":
        """Create a SearchConfig from environment variables.

        Returns:
            A SearchConfig populated from env vars.

        Raises:
            ValueError: If required env vars are missing.
        """

        lambda_function_name = (os.getenv("SEARCH_LAMBDA_FUNCTION_NAME") or "").strip()
        if not lambda_function_name:
            raise ValueError("Missing required environment variable: SEARCH_LAMBDA_FUNCTION_NAME")

        index_bucket = (os.getenv("SEARCH_INDEX_BUCKET") or "").strip()
        if not index_bucket:
            raise ValueError("Missing required environment variable: SEARCH_INDEX_BUCKET")

        index_prefix = (os.getenv("SEARCH_INDEX_PREFIX") or "").strip() or "search-index/"

        region = (
            os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
        )

        bm25_weight = float(os.getenv("BM25_WEIGHT", "0.3"))
        vector_weight = float(os.getenv("VECTOR_WEIGHT", "0.7"))

        timeout_raw = (os.getenv("SEARCH_LAMBDA_TIMEOUT_SECONDS") or "").strip()
        lambda_timeout_seconds = 30
        if timeout_raw:
            try:
                lambda_timeout_seconds = int(timeout_raw)
            except ValueError as exc:
                raise ValueError("SEARCH_LAMBDA_TIMEOUT_SECONDS must be an integer") from exc

        return SearchConfig(
            lambda_function_name=lambda_function_name,
            index_bucket=index_bucket,
            index_prefix=index_prefix,
            region=region,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            lambda_timeout_seconds=lambda_timeout_seconds,
        )