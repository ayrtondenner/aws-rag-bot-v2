from __future__ import annotations

from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.llm_agent import ToolUnion

from app.models.search import (
    DocumentExistsResponse,
    IndexDocumentResponse,
    IndexedDocumentsResponse,
    IndexStatsResponse,
    SearchResponse,
)
from app.services.dependencies import get_search_service as get_search_service_dependency
from app.services.search_service import SearchService

from shared import transfer_to_root


def _get_search_service() -> SearchService:
    """Build a SearchService using the same factory path as the FastAPI dependency."""

    return get_search_service_dependency()


async def search_query(
    *,
    query: str,
    size: int = 10,
    search_type: str = "hybrid",
) -> SearchResponse:
    """Search indexed SageMaker documents using hybrid (BM25 + vector), text-only, or vector-only.

    Args:
        query: The search query text.
        size: Maximum number of results to return (1-100).
        search_type: One of 'hybrid', 'text', or 'vector'.

    Returns:
        JSON with {query, search_type, total_hits, hits:[{doc_id, score, filename, content}, ...]}.
    """

    svc = _get_search_service()
    return await svc.search(query=query, size=size, search_type=search_type)  # type: ignore[arg-type]


async def search_index_document(
    *,
    filename: str,
    content: str,
) -> IndexDocumentResponse:
    """Index a single document (splits into chunks automatically).

    Skips indexing if the filename is already present in the index.

    Args:
        filename: Source document filename (e.g. 'amazon-sagemaker-toolkits.md').
        content: Full text content of the document.

    Returns:
        JSON with {filename, chunk_count, doc_ids, skipped}.
    """

    svc = _get_search_service()
    return await svc.index_document(filename=filename, content=content)


async def search_document_exists(*, filename: str) -> DocumentExistsResponse:
    """Check whether a document filename is already indexed.

    Args:
        filename: The document filename to check.

    Returns:
        JSON with {filename, exists}.
    """

    svc = _get_search_service()
    exists = await svc.document_exists(filename=filename)
    return DocumentExistsResponse(filename=filename, exists=exists)


async def search_list_indexed_documents() -> IndexedDocumentsResponse:
    """List all unique document filenames currently indexed.

    Returns:
        JSON with {count, filenames:[...]}.
    """

    svc = _get_search_service()
    filenames = await svc.list_indexed_documents()
    return IndexedDocumentsResponse(count=len(filenames), filenames=filenames)


async def search_delete_document(*, doc_id: str) -> dict[str, bool]:
    """Delete a single document chunk by its ID.

    Args:
        doc_id: The document chunk ID to delete.

    Returns:
        JSON with {deleted: true/false}.
    """

    svc = _get_search_service()
    deleted = await svc.delete_document(doc_id=doc_id)
    return {"deleted": deleted}


async def search_delete_by_filename(*, filename: str) -> dict[str, int]:
    """Delete all chunks for a given source filename.

    Args:
        filename: The source document filename whose chunks should be removed.

    Returns:
        JSON with {deleted_count: int}.
    """

    svc = _get_search_service()
    count = await svc.delete_documents_by_filename(filename=filename)
    return {"deleted_count": count}


async def search_get_index_stats() -> IndexStatsResponse:
    """Get basic statistics for the search index (document count, status).

    Returns:
        JSON with {index_name, doc_count, status}.
    """

    svc = _get_search_service()
    return await svc.get_index_stats()


def build_search_tools() -> list[ToolUnion]:
    """Build the list of search ADK tools for the agent."""

    return [
        FunctionTool(search_query),
        FunctionTool(search_index_document),
        FunctionTool(search_document_exists),
        FunctionTool(search_list_indexed_documents),
        FunctionTool(search_get_index_stats),
        FunctionTool(search_delete_document),
        FunctionTool(search_delete_by_filename),
        FunctionTool(transfer_to_root),
    ]
