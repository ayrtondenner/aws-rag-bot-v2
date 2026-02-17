from __future__ import annotations

from typing import Annotated
from pydantic import Field

from app.models.opensearch import (
    DocumentExistsResponse,
    IndexDocumentResponse,
    IndexedDocumentsResponse,
    IndexStatsResponse,
    SearchResponse,
)
import shared.opensearch_tools as shared_os

from mcp_server import mcp


@mcp.resource(
    name="opensearch_query",
    description="Search indexed SageMaker documents using hybrid (BM25 + neural), text-only, or vector-only.",
    uri="opensearch://query/{?query,size,search_type}",
)
async def opensearch_query(
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

    return await shared_os.opensearch_query(query=query, size=size, search_type=search_type)


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

    return await shared_os.opensearch_index_document(filename=filename, content=content)


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

    return await shared_os.opensearch_document_exists(filename=filename)


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

    return await shared_os.opensearch_list_indexed_documents()


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

    return await shared_os.opensearch_get_index_stats()
