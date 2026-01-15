from __future__ import annotations

from typing import Any, Optional

from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.llm_agent import ToolUnion
from google.adk.tools.transfer_to_agent_tool import transfer_to_agent
from google.adk.tools.tool_context import ToolContext

from app.models.document import LocalDocumentContentResponse, LocalDocumentsResponse
from app.models.s3 import BucketExistsResponse, FileListResponse, S3FileContentResponse
from app.services.config import S3Config
from app.services.dependencies import get_document_service as get_document_service_dependency
from app.services.dependencies import get_s3_service as get_s3_service_dependency
from app.services.document_service import DocumentService
from app.services.s3_service import S3Service

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="aws-rag-bot-mcp",
    instructions=(
        "Shared tool server for AWS RAG Bot. "
        "Exposes S3 and local-document utilities designed to be reused by both "
        "ADK agents and the MCP server."
    ),
)

DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME = S3Config.from_env().bucket_name.strip()

def _get_s3_service() -> S3Service:
    if not DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME:
        raise ValueError("bucket_name must be provided")

    # Use the same constructor path as the FastAPI dependency for the default bucket.
    return get_s3_service_dependency()


def _get_document_service() -> DocumentService:
    # Use the same constructor path as the FastAPI dependency.
    return get_document_service_dependency()

def transfer_to_root(tool_context: ToolContext) -> None:
    """Transfer control back to the root agent.

    Use this when the user's request is not about the sub-agent responsibility.
    """

    transfer_to_agent("root_agent", tool_context)

@mcp.tool(
    name="s3_bucket_exists",
    description="Check whether an S3 bucket exists and is accessible.",
)
async def s3_bucket_exists(*, bucket_name: str = DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME) -> dict[str, Any]:
    """Check whether an S3 bucket exists and is accessible.

    Args:
        bucket_name: Bucket name to check. If omitted, defaults to the SageMaker docs bucket

    Returns:
        JSON with {bucket_name, exists}.
    """

    s3 = _get_s3_service()
    exists = await s3.bucket_exists(bucket_name=bucket_name)
    return BucketExistsResponse(bucket_name=bucket_name, exists=exists).model_dump()

@mcp.tool(
    name="s3_list_bucket_files",
    description="List files (object keys) in the configured S3 bucket.",
)
async def s3_list_bucket_files(
    *,
    prefix: Optional[str] = None,
    max_keys: int = 1000,
) -> dict[str, Any]:
    """List files (object keys) in an S3 bucket.

    Args:
        prefix: Optional key prefix filter.
        max_keys: Max number of keys to return (S3 ListObjectsV2 MaxKeys).

    Returns:
        JSON with {count, files:[{key,size,last_modified,etag}, ...]}.
    """

    s3 = _get_s3_service()
    files = await s3.list_files(prefix=prefix, max_keys=max_keys)
    return FileListResponse(count=len(files), files=files).model_dump()

@mcp.tool(
    name="s3_get_file_content",
    description="Fetch the text content of an S3 object.",
)
async def s3_get_file_content(
    *,
    key: str,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """Fetch the content of an S3 object.

    Args:
        key: The object key to fetch.
        encoding: Text decoding encoding.

    Returns:
        JSON with {filename, content}.
    """

    s3 = _get_s3_service()
    content = await s3.get_file_content(key=key, encoding=encoding)
    return S3FileContentResponse(filename=key, content=content).model_dump()

@mcp.tool(
    name="list_local_sagemaker_docs",
    description="List files in the local sagemaker-docs folder.",
)
async def list_local_sagemaker_docs() -> dict[str, Any]:
    """List files in the local `sagemaker-docs` folder.

    Returns:
        JSON with {count, documents:[filename,...]}.
    """

    documents = _get_document_service()
    result = documents.list_local_sagemaker_docs()
    return LocalDocumentsResponse.model_validate(result).model_dump()


@mcp.tool(
    name="get_local_sagemaker_doc_content",
    description="Get the text content of a local file in the sagemaker-docs folder by filename.",
)
async def get_local_sagemaker_doc_content(*, filename: str) -> dict[str, Any]:
    """Read a local doc file content.

    Returns:
        JSON with {filename, content}.
    """

    documents = _get_document_service()
    content = documents.get_local_sagemaker_doc_content(filename=filename)
    return LocalDocumentContentResponse(filename=filename, content=content).model_dump()


def build_s3_tools() -> list[ToolUnion]:
    return [
        FunctionTool(s3_bucket_exists),
        FunctionTool(s3_list_bucket_files),
        FunctionTool(s3_get_file_content),
        FunctionTool(transfer_to_root),
    ]


def build_document_tools() -> list[ToolUnion]:
    return [
        FunctionTool(list_local_sagemaker_docs),
        FunctionTool(get_local_sagemaker_doc_content),
        FunctionTool(transfer_to_root),
    ]

