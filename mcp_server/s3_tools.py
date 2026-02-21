from __future__ import annotations

from typing import Optional, Annotated
from pydantic import Field

from app.models.s3 import BucketExistsResponse, FileListResponse, S3FileContentResponse
from shared import DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME
import shared.s3_tools as shared_s3

from mcp_server import mcp


@mcp.resource(
    name="s3_bucket_exists",
    description="Check whether an S3 bucket exists and is accessible.",
    uri="s3://bucket_exists/{?bucket_name}",
)
async def s3_bucket_exists(*,
    bucket_name: Annotated[
        str,
        Field(
            default=DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME,
            description="The name of the S3 bucket to check. If omitted, defaults to the SageMaker docs bucket."
        )
    ] = DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME
) -> BucketExistsResponse:
    """Check whether an S3 bucket exists and is accessible.

    Args:
        bucket_name: Bucket name to check. If omitted, defaults to the SageMaker docs bucket

    Returns:
        JSON with {bucket_name, exists}.
    """

    return await shared_s3.s3_bucket_exists(bucket_name=bucket_name)


@mcp.resource(
    name="s3_list_bucket_files",
    description="List files (object keys) in the configured S3 bucket.",
    uri="s3://list_bucket_files/{?prefix,max_keys}",
    annotations={"prefix": "Optional key prefix filter.", "max_keys": "Max number of keys to return (S3 ListObjectsV2 MaxKeys)."},
)
async def s3_list_bucket_files(
    *,
    prefix: Annotated[Optional[str], Field(default=None, description="Optional key prefix filter.")] = None,
    max_keys: Annotated[
        Optional[int],
        Field(default=1000, ge=1, le=1000, description="Max number of keys to return (S3 ListObjectsV2 MaxKeys)."),
    ] = 1000,
) -> FileListResponse:
    """List files (object keys) in an S3 bucket.

    Args:
        prefix: Optional key prefix filter.
        max_keys: Max number of keys to return (S3 ListObjectsV2 MaxKeys).

    Returns:
        JSON with {count, files:[{key,size,last_modified,etag}, ...]}.
    """

    return await shared_s3.s3_list_bucket_files(prefix=prefix, max_keys=max_keys)


@mcp.resource(
    name="s3_get_file_content",
    description="Fetch the text content of an S3 object.",
    uri="s3://get_file_content/{key}{?encoding}",
)
async def s3_get_file_content(
    *,
    key: Annotated[
        str,
        Field(..., description="The S3 object key to fetch."),
    ],
    encoding: Annotated[
        str,
        Field(default="utf-8", description="Text decoding encoding."),
    ] = "utf-8",
) -> S3FileContentResponse:
    """Fetch the content of an S3 object.

    Args:
        key: The object key to fetch.
        encoding: Text decoding encoding.

    Returns:
        JSON with {filename, content}.
    """

    return await shared_s3.s3_get_file_content(key=key, encoding=encoding)
