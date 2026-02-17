from __future__ import annotations

from typing import Optional

from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.llm_agent import ToolUnion
from google.adk.tools.transfer_to_agent_tool import transfer_to_agent
from google.adk.tools.tool_context import ToolContext

from app.models.document import LocalDocumentContentResponse, LocalDocumentsResponse
from app.models.opensearch import (
    DocumentExistsResponse,
    IndexDocumentResponse,
    IndexedDocumentsResponse,
    IndexStatsResponse,
    SearchResponse,
)
from app.models.s3 import BucketExistsResponse, FileListResponse, S3FileContentResponse
from app.services.dependencies import get_document_service as get_document_service_dependency
from app.services.dependencies import get_opensearch_service as get_opensearch_service_dependency
from app.services.dependencies import get_s3_service as get_s3_service_dependency
from app.services.document_service import DocumentService
from app.services.opensearch_service import OpenSearchService
from app.services.s3_service import S3Service

# TODO: import this from S3 config instead
DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME = "senior-sagemaker-assessment-bucket"

def _get_s3_service() -> S3Service:
    if not DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME:
        raise ValueError("bucket_name must be provided")

    # Use the same constructor path as the FastAPI dependency for the default bucket.
    return get_s3_service_dependency()


def _get_document_service() -> DocumentService:
    # Use the same constructor path as the FastAPI dependency.
    return get_document_service_dependency()


def _get_opensearch_service() -> OpenSearchService:
    # Use the same constructor path as the FastAPI dependency.
    return get_opensearch_service_dependency()

def transfer_to_root(tool_context: ToolContext) -> None:
    """Transfer control back to the root agent.

    Use this when the user's request is not about the sub-agent responsibility.
    """

    transfer_to_agent("root_agent", tool_context)

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

async def list_local_sagemaker_docs() -> LocalDocumentsResponse:
    """List files in the local `sagemaker-docs` folder.

    Returns:
        JSON with {count, documents:[filename,...]}.
    """

    documents = _get_document_service()
    result = documents.list_local_sagemaker_docs()
    return LocalDocumentsResponse.model_validate(result)

async def get_local_sagemaker_doc_content(*, filename: str) -> LocalDocumentContentResponse:
    """Read a local doc file content.

    Returns:
        JSON with {filename, content}.
    """

    documents = _get_document_service()
    content = documents.get_local_sagemaker_doc_content(filename=filename)
    return LocalDocumentContentResponse(filename=filename, content=content)


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


# ---------------------------------------------------------------------------
# OpenSearch tools
# ---------------------------------------------------------------------------


async def opensearch_search(
    *,
    query: str,
    size: int = 10,
    search_type: str = "hybrid",
) -> SearchResponse:
    """Search indexed SageMaker documents using hybrid (BM25 + neural), text-only, or vector-only.

    Args:
        query: The search query text.
        size: Maximum number of results to return (1-100).
        search_type: One of 'hybrid', 'text', or 'vector'.

    Returns:
        JSON with {query, search_type, total_hits, hits:[{doc_id, score, filename, content}, ...]}.
    """

    opensearch = _get_opensearch_service()
    return await opensearch.search(query=query, size=size, search_type=search_type)  # type: ignore[arg-type]


async def opensearch_index_document(
    *,
    filename: str,
    content: str,
) -> IndexDocumentResponse:
    """Index a single document into OpenSearch (splits into chunks automatically).

    Skips indexing if the filename is already present in the index.

    Args:
        filename: Source document filename (e.g. 'amazon-sagemaker-toolkits.md').
        content: Full text content of the document.

    Returns:
        JSON with {filename, chunk_count, doc_ids, skipped}.
    """

    opensearch = _get_opensearch_service()
    return await opensearch.index_document(filename=filename, content=content)


async def opensearch_document_exists(*, filename: str) -> DocumentExistsResponse:
    """Check whether a document filename is already indexed in OpenSearch.

    Args:
        filename: The document filename to check.

    Returns:
        JSON with {filename, exists}.
    """

    opensearch = _get_opensearch_service()
    exists = await opensearch.document_exists(filename=filename)
    return DocumentExistsResponse(filename=filename, exists=exists)


async def opensearch_list_indexed_documents() -> IndexedDocumentsResponse:
    """List all unique document filenames currently indexed in OpenSearch.

    Returns:
        JSON with {count, filenames:[...]}.
    """

    opensearch = _get_opensearch_service()
    filenames = await opensearch.list_indexed_documents()
    return IndexedDocumentsResponse(count=len(filenames), filenames=filenames)


async def opensearch_get_index_stats() -> IndexStatsResponse:
    """Get basic statistics for the OpenSearch index (document count, status).

    Returns:
        JSON with {index_name, doc_count, status}.
    """

    opensearch = _get_opensearch_service()
    return await opensearch.get_index_stats()


def build_opensearch_tools() -> list[ToolUnion]:
    """Build the list of OpenSearch ADK tools for the agent."""

    return [
        FunctionTool(opensearch_search),
        FunctionTool(opensearch_index_document),
        FunctionTool(opensearch_document_exists),
        FunctionTool(opensearch_list_indexed_documents),
        FunctionTool(opensearch_get_index_stats),
        FunctionTool(transfer_to_root),
    ]