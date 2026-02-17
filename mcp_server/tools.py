from __future__ import annotations

from typing import Optional, Annotated
from pydantic import Field

from app.models.document import LocalDocumentContentResponse, LocalDocumentsResponse
from app.models.opensearch import (
    DocumentExistsResponse,
    IndexDocumentResponse,
    IndexedDocumentsResponse,
    IndexStatsResponse,
    SearchResponse,
)
from app.models.s3 import BucketExistsResponse, FileListResponse, S3FileContentResponse
import shared.tools as shared_tools

from fastmcp import FastMCP

# TODO: rethink if mcp object should be in this file
mcp = FastMCP(
    name="aws-rag-bot-mcp",
    instructions=(
        "Shared tool server for AWS RAG Bot. "
        "Exposes S3 and local-document utilities designed to be reused by both "
        "ADK agents and the MCP server."
    ),
)

# TODO: Refactor tools to divide between: S3, local docs and OpenSearch

# TODO: think of tools examples, right now we only have resources
# Maybe tools/functions to upload files on S3/folder on the fly?
@mcp.resource(
    name="s3_bucket_exists",
    description="Check whether an S3 bucket exists and is accessible.",
    uri="s3://bucket_exists/{?bucket_name}",
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

    return await shared_tools.s3_list_bucket_files(prefix=prefix, max_keys=max_keys)

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

    return await shared_tools.s3_get_file_content(key=key, encoding=encoding)

@mcp.resource(
    name="list_local_sagemaker_docs",
    description="List files in the local sagemaker-docs folder.",
    uri="local://sagemaker-docs/",
)
async def list_local_sagemaker_docs() -> LocalDocumentsResponse:
    """List files in the local `sagemaker-docs` folder.

    Returns:
        JSON with {count, documents:[filename,...]}.
    """

    return await shared_tools.list_local_sagemaker_docs()


@mcp.resource(
    name="get_local_sagemaker_doc_content",
    description="Get the text content of a local file in the sagemaker-docs folder by filename.",
    uri="local://sagemaker-docs/{filename}",
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


# ---------------------------------------------------------------------------
# OpenSearch resources
# ---------------------------------------------------------------------------


@mcp.resource(
    name="opensearch_search",
    description="Search indexed SageMaker documents using hybrid (BM25 + neural), text-only, or vector-only.",
    uri="opensearch://search/{?query,size,search_type}",
)
async def opensearch_search(
    *,
    query: Annotated[str, Field(..., description="The search query text.")],
    size: Annotated[int, Field(default=10, ge=1, le=100, description="Maximum number of results.")] = 10,
    search_type: Annotated[
        str,
        Field(default="hybrid", description="Search strategy: 'hybrid', 'text', or 'vector'."),
    ] = "hybrid",
) -> SearchResponse:
    """Search indexed documents using hybrid (BM25 + neural), text-only, or vector-only.

    Args:
        query: The search query text.
        size: Maximum number of results to return.
        search_type: One of 'hybrid', 'text', or 'vector'.

    Returns:
        JSON with {query, search_type, total_hits, hits:[...]}.
    """

    return await shared_tools.opensearch_search(query=query, size=size, search_type=search_type)


@mcp.resource(
    name="opensearch_index_document",
    description="Index a single document into OpenSearch (auto-chunks, skips duplicates).",
    uri="opensearch://index/{?filename,content}",
)
async def opensearch_index_document(
    *,
    filename: Annotated[str, Field(..., description="Source document filename.")],
    content: Annotated[str, Field(..., description="Full text content of the document.")],
) -> IndexDocumentResponse:
    """Index a single document into OpenSearch.

    Args:
        filename: Source document filename.
        content: Full text content.

    Returns:
        JSON with {filename, chunk_count, doc_ids, skipped}.
    """

    return await shared_tools.opensearch_index_document(filename=filename, content=content)


@mcp.resource(
    name="opensearch_document_exists",
    description="Check whether a document filename is already indexed in OpenSearch.",
    uri="opensearch://document_exists/{filename}",
)
async def opensearch_document_exists(
    *,
    filename: Annotated[str, Field(..., description="Document filename to check.")],
) -> DocumentExistsResponse:
    """Check whether a document filename is already indexed.

    Args:
        filename: The document filename to check.

    Returns:
        JSON with {filename, exists}.
    """

    return await shared_tools.opensearch_document_exists(filename=filename)


@mcp.resource(
    name="opensearch_list_indexed_documents",
    description="List all unique document filenames currently indexed in OpenSearch.",
    uri="opensearch://documents/",
)
async def opensearch_list_indexed_documents() -> IndexedDocumentsResponse:
    """List all indexed document filenames.

    Returns:
        JSON with {count, filenames}.
    """

    return await shared_tools.opensearch_list_indexed_documents()


@mcp.resource(
    name="opensearch_get_index_stats",
    description="Get basic statistics for the OpenSearch index.",
    uri="opensearch://index/stats",
)
async def opensearch_get_index_stats() -> IndexStatsResponse:
    """Get index-level statistics (document count, status).

    Returns:
        JSON with {index_name, doc_count, status}.
    """

    return await shared_tools.opensearch_get_index_stats()