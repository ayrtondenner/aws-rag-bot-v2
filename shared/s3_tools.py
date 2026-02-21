from __future__ import annotations

from typing import Optional

from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.llm_agent import ToolUnion

from app.models.s3 import BucketExistsResponse, FileListResponse, S3FileContentResponse
from app.services.dependencies import get_s3_service as get_s3_service_dependency
from app.services.s3_service import S3Service

from shared import DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME, transfer_to_root


def _get_s3_service() -> S3Service:
    """Build an S3Service using the same factory path as the FastAPI dependency."""

    if not DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME:
        raise ValueError("bucket_name must be provided")

    return get_s3_service_dependency()


async def s3_bucket_exists(*, bucket_name: str = DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME) -> BucketExistsResponse:
    """Check whether an S3 bucket exists and is accessible.

    Args:
        bucket_name: Bucket name to check. If omitted, defaults to the SageMaker docs bucket

    Returns:
        JSON with {bucket_name, exists}.
    """

    s3 = _get_s3_service()
    exists = await s3.bucket_exists(bucket_name=bucket_name)
    return BucketExistsResponse(bucket_name=bucket_name, exists=exists)


async def s3_list_bucket_files(
    *,
    prefix: Optional[str] = None,
    max_keys: Optional[int] = 1000,
) -> FileListResponse:
    """List files (object keys) in an S3 bucket.

    Args:
        prefix: Optional key prefix filter.
        max_keys: Max number of keys to return (S3 ListObjectsV2 MaxKeys).

    Returns:
        JSON with {count, files:[{key,size,last_modified,etag}, ...]}.
    """

    s3 = _get_s3_service()
    files = await s3.list_files(prefix=prefix, max_keys=max_keys)
    return FileListResponse(count=len(files), files=files)


async def s3_get_file_content(
    *,
    key: str,
    encoding: str = "utf-8",
) -> S3FileContentResponse:
    """Fetch the content of an S3 object.

    Args:
        key: The object key to fetch.
        encoding: Text decoding encoding.

    Returns:
        JSON with {filename, content}.
    """

    s3 = _get_s3_service()
    content = await s3.get_file_content(key=key, encoding=encoding)
    return S3FileContentResponse(filename=key, content=content)


def build_s3_tools() -> list[ToolUnion]:
    """Build the list of S3 ADK tools for the agent."""

    return [
        FunctionTool(s3_bucket_exists),
        FunctionTool(s3_list_bucket_files),
        FunctionTool(s3_get_file_content),
        FunctionTool(transfer_to_root),
    ]
