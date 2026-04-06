from __future__ import annotations

from typing import Annotated
from pydantic import Field

from app.models.search import (
    DocumentExistsResponse,
    IndexDocumentResponse,
    IndexedDocumentsResponse,
    IndexStatsResponse,
    SearchResponse,
)
import shared.search_tools as shared_search

from mcp_server import mcp


@mcp.resource(
    name="search_query",
    description="Search indexed SageMaker documents using hybrid (BM25 + vector), text-only, or vector-only.",
    uri="search://query/{query}",
)
async def search_query(
    *,
    query: Annotated[str, Field(..., description="The search query text.")],
    size: Annotated[int, Field(default=10, ge=1, le=100, description="Maximum number of results.")] = 10,
    search_type: Annotated[
        str,
        Field(default="hybrid", description="Search strategy: 'hybrid', 'text', or 'vector'."),
    ] = "hybrid",
) -> SearchResponse:
    """Search indexed documents using hybrid (BM25 + vector), text-only, or vector-only.

    Args:
        query: The search query text.
        size: Maximum number of results to return.
        search_type: One of 'hybrid', 'text', or 'vector'.

    Returns:
        JSON with {query, search_type, total_hits, hits:[...]}.
    """

    return await shared_search.search_query(query=query, size=size, search_type=search_type)


@mcp.tool(
    name="search_index_document",
    description="Index a single document (auto-chunks, skips duplicates).",
)
async def search_index_document(
    *,
    filename: Annotated[str, Field(..., description="Source document filename.")],
    content: Annotated[str, Field(..., description="Full text content of the document.")],
) -> IndexDocumentResponse:
    """Index a single document.

    Args:
        filename: Source document filename.
        content: Full text content.

    Returns:
        JSON with {filename, chunk_count, doc_ids, skipped}.
    """

    return await shared_search.search_index_document(filename=filename, content=content)


@mcp.resource(
    name="search_document_exists",
    description="Check whether a document filename is already indexed.",
    uri="search://document_exists/{filename}",
)
async def search_document_exists(
    *,
    filename: Annotated[str, Field(..., description="Document filename to check.")],
) -> DocumentExistsResponse:
    """Check whether a document filename is already indexed.

    Args:
        filename: The document filename to check.

    Returns:
        JSON with {filename, exists}.
    """

    return await shared_search.search_document_exists(filename=filename)


@mcp.resource(
    name="search_list_indexed_documents",
    description="List all unique document filenames currently indexed.",
    uri="search://documents/",
)
async def search_list_indexed_documents() -> IndexedDocumentsResponse:
    """List all indexed document filenames.

    Returns:
        JSON with {count, filenames}.
    """

    return await shared_search.search_list_indexed_documents()


@mcp.resource(
    name="search_get_index_stats",
    description="Get basic statistics for the search index.",
    uri="search://index/stats",
)
async def search_get_index_stats() -> IndexStatsResponse:
    """Get index-level statistics (document count, status).

    Returns:
        JSON with {index_name, doc_count, status}.
    """

    return await shared_search.search_get_index_stats()


@mcp.tool(
    name="search_delete_document",
    description="Delete a single document chunk by its ID.",
)
async def search_delete_document(
    *,
    doc_id: Annotated[str, Field(..., description="The document chunk ID to delete.")],
) -> dict[str, bool]:
    """Delete a single document chunk by its ID.

    Args:
        doc_id: The document chunk ID.

    Returns:
        JSON with {deleted: true/false}.
    """

    return await shared_search.search_delete_document(doc_id=doc_id)


@mcp.tool(
    name="search_delete_by_filename",
    description="Delete all indexed chunks for a given source filename.",
)
async def search_delete_by_filename(
    *,
    filename: Annotated[str, Field(..., description="Source document filename whose chunks to delete.")],
) -> dict[str, int]:
    """Delete all chunks for a given source filename.

    Args:
        filename: The source document filename.

    Returns:
        JSON with {deleted_count: int}.
    """

    return await shared_search.search_delete_by_filename(filename=filename)
