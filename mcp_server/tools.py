from __future__ import annotations

from typing import Optional, Annotated
from pydantic import Field

from app.models.document import LocalDocumentContentResponse, LocalDocumentsResponse
from app.models.s3 import BucketExistsResponse, FileListResponse, S3FileContentResponse
import shared.tools as shared_tools

from fastmcp import FastMCP

mcp = FastMCP(
    name="aws-rag-bot-mcp",
    instructions=(
        "Shared tool server for AWS RAG Bot. "
        "Exposes S3 and local-document utilities designed to be reused by both "
        "ADK agents and the MCP server."
    ),
)

@mcp.tool(
    name="s3_bucket_exists",
    description="Check whether an S3 bucket exists and is accessible.",
)
async def s3_bucket_exists(*,
    bucket_name: Annotated[
        str,
        Field(
            default=shared_tools.DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME,
            description="The name of the S3 bucket to check. If omitted, defaults to the SageMaker docs bucket."
        )
    ] = shared_tools.DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME
) -> BucketExistsResponse:
    """Check whether an S3 bucket exists and is accessible.

    Args:
        bucket_name: Bucket name to check. If omitted, defaults to the SageMaker docs bucket

    Returns:
        JSON with {bucket_name, exists}.
    """

    return await shared_tools.s3_bucket_exists(bucket_name=bucket_name)

@mcp.tool(
    name="s3_list_bucket_files",
    description="List files (object keys) in the configured S3 bucket.",
)
async def s3_list_bucket_files(
    *,
    prefix: Annotated[Optional[str], Field(default=None, description="Optional key prefix filter.")] = None,
    max_keys: Annotated[
        int,
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

    return await shared_tools.s3_list_bucket_files(prefix=prefix, max_keys=max_keys)

@mcp.tool(
    name="s3_get_file_content",
    description="Fetch the text content of an S3 object.",
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

    return await shared_tools.s3_get_file_content(key=key, encoding=encoding)

@mcp.tool(
    name="list_local_sagemaker_docs",
    description="List files in the local sagemaker-docs folder.",
)
async def list_local_sagemaker_docs() -> LocalDocumentsResponse:
    """List files in the local `sagemaker-docs` folder.

    Returns:
        JSON with {count, documents:[filename,...]}.
    """

    return await shared_tools.list_local_sagemaker_docs()


@mcp.tool(
    name="get_local_sagemaker_doc_content",
    description="Get the text content of a local file in the sagemaker-docs folder by filename.",
)
async def get_local_sagemaker_doc_content(
    *,
    filename: Annotated[
        str,
        Field(..., description="Filename of the local doc in the sagemaker-docs folder to read."),
    ],
) -> LocalDocumentContentResponse:
    """Read a local doc file content.

    Returns:
        JSON with {filename, content}.
    """

    return await shared_tools.get_local_sagemaker_doc_content(filename=filename)