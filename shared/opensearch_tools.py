from __future__ import annotations

from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.llm_agent import ToolUnion

from app.models.opensearch import (
    DocumentExistsResponse,
    IndexDocumentResponse,
    IndexedDocumentsResponse,
    IndexStatsResponse,
    SearchResponse,
)
from app.services.dependencies import get_opensearch_service as get_opensearch_service_dependency
from app.services.opensearch_service import OpenSearchService

from shared import transfer_to_root


def _get_opensearch_service() -> OpenSearchService:
    """Build an OpenSearchService using the same factory path as the FastAPI dependency."""

    return get_opensearch_service_dependency()


async def opensearch_query(
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
        FunctionTool(opensearch_query),
        FunctionTool(opensearch_index_document),
        FunctionTool(opensearch_document_exists),
        FunctionTool(opensearch_list_indexed_documents),
        FunctionTool(opensearch_get_index_stats),
        FunctionTool(transfer_to_root),
    ]
