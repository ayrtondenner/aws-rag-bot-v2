from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from starlette import status

from app.models.search import (
    BulkIndexRequest,
    BulkIndexResponse,
    DeleteByFilenameResponse,
    DocumentExistsResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    IndexedDocumentsResponse,
    IndexStatsResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.dependencies import get_document_service, get_search_service
from app.services.document_service import DocumentService
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


# ------------------------------------------------------------------
# Indexing
# ------------------------------------------------------------------


@router.post(
    "/index",
    response_model=IndexDocumentResponse,
    summary="Index a single document",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "filename": "amazon-sagemaker-toolkits.md",
                        "chunk_count": 5,
                        "doc_ids": ["abc123", "def456", "ghi789", "jkl012", "mno345"],
                        "skipped": False,
                    }
                }
            }
        },
        400: {"description": "Validation error (blank filename / content)"},
    },
)
async def index_document(
    payload: IndexDocumentRequest = Body(
        ...,
        examples=[
            {
                "filename": "amazon-sagemaker-toolkits.md",
                "content": "# Using the SageMaker Training and Inference Toolkits ...",
            }
        ],
    ),
    chunk_size: int = Query(default=500, ge=1, description="Target chunk size in characters."),
    chunk_overlap: int = Query(default=50, ge=0, description="Overlap between chunks."),
    search_svc: SearchService = Depends(get_search_service),
) -> IndexDocumentResponse:
    """Index a single document into the search index.

    The document is split into overlapping chunks and each chunk is indexed
    separately. If the filename is already indexed, the operation is skipped.
    """

    try:
        return await search_svc.index_document(
            filename=payload.filename,
            content=payload.content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/bulk-index",
    response_model=BulkIndexResponse,
    summary="Bulk-index multiple documents",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "total_chunks": 12,
                        "indexed_count": 2,
                        "skipped_count": 1,
                        "results": [
                            {"filename": "doc-a.md", "chunk_count": 5, "doc_ids": ["a1", "a2", "a3", "a4", "a5"], "skipped": False},
                            {"filename": "doc-b.md", "chunk_count": 7, "doc_ids": ["b1", "b2", "b3", "b4", "b5", "b6", "b7"], "skipped": False},
                            {"filename": "doc-c.md", "chunk_count": 0, "doc_ids": [], "skipped": True},
                        ],
                    }
                }
            }
        },
    },
)
async def bulk_index_documents(
    payload: BulkIndexRequest = Body(...),
    search_svc: SearchService = Depends(get_search_service),
) -> BulkIndexResponse:
    """Bulk-index multiple documents. Already-indexed filenames are skipped."""

    try:
        return await search_svc.bulk_index_documents(
            documents=payload.documents,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            max_concurrency=payload.max_concurrency,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/index-local-docs",
    response_model=BulkIndexResponse,
    summary="Index all local sagemaker-docs",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "total_chunks": 100,
                        "indexed_count": 50,
                        "skipped_count": 286,
                        "results": [],
                    }
                }
            }
        },
    },
)
async def index_local_docs(
    chunk_size: int = Query(default=500, ge=1, description="Target chunk size in characters."),
    chunk_overlap: int = Query(default=50, ge=0, description="Overlap between chunks."),
    max_concurrency: int = Query(default=5, ge=1, le=20, description="Maximum documents indexed in parallel."),
    search_svc: SearchService = Depends(get_search_service),
    documents: DocumentService = Depends(get_document_service),
) -> BulkIndexResponse:
    """Read ALL files from the local ``sagemaker-docs/`` folder and bulk-index them.

    Already-indexed files are automatically skipped, making this endpoint
    safe to call repeatedly (idempotent).
    """

    local = documents.list_local_sagemaker_docs()
    filenames: list[str] = local.get("documents", [])

    docs: list[IndexDocumentRequest] = []
    for fname in filenames:
        try:
            content = documents.get_local_sagemaker_doc_content(filename=fname)
            docs.append(IndexDocumentRequest(filename=fname, content=content))
        except (FileNotFoundError, ValueError):
            continue

    if not docs:
        return BulkIndexResponse(total_chunks=0, indexed_count=0, skipped_count=0, results=[])

    return await search_svc.bulk_index_documents(
        documents=docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_concurrency=max_concurrency,
    )


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search documents (hybrid by default)",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "query": "SageMaker endpoint configuration",
                        "search_type": "hybrid",
                        "total_hits": 2,
                        "hits": [
                            {
                                "doc_id": "abc123",
                                "score": 0.85,
                                "filename": "aws-resource-sagemaker-endpointconfig.md",
                                "content": "# AWS::SageMaker::EndpointConfig ...",
                            },
                            {
                                "doc_id": "def456",
                                "score": 0.72,
                                "filename": "aws-resource-sagemaker-endpoint.md",
                                "content": "# AWS::SageMaker::Endpoint ...",
                            },
                        ],
                    }
                }
            }
        },
        400: {"description": "Empty query"},
    },
)
async def search(
    payload: SearchRequest = Body(
        ...,
        examples=[
            {"query": "SageMaker endpoint configuration", "size": 5, "search_type": "hybrid"},
        ],
    ),
    search_svc: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Search indexed documents using hybrid (BM25 + vector), text-only, or vector-only."""

    try:
        return await search_svc.search(
            query=payload.query,
            size=payload.size,
            search_type=payload.search_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


# ------------------------------------------------------------------
# Document queries
# ------------------------------------------------------------------


@router.get(
    "/index/stats",
    response_model=IndexStatsResponse,
    summary="Get index statistics",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "index_name": "sagemaker-docs",
                        "doc_count": 5000,
                        "status": "available",
                    }
                }
            }
        },
    },
)
async def index_stats(
    search_svc: SearchService = Depends(get_search_service),
) -> IndexStatsResponse:
    """Return basic statistics for the search index."""

    return await search_svc.get_index_stats()


@router.get(
    "/documents",
    response_model=IndexedDocumentsResponse,
    summary="List indexed document filenames",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "count": 3,
                        "filenames": [
                            "amazon-sagemaker-toolkits.md",
                            "aws-resource-sagemaker-endpoint.md",
                            "aws-resource-sagemaker-endpointconfig.md",
                        ],
                    }
                }
            }
        },
    },
)
async def list_indexed_documents(
    search_svc: SearchService = Depends(get_search_service),
) -> IndexedDocumentsResponse:
    """Return a deduplicated list of all indexed document filenames."""

    filenames = await search_svc.list_indexed_documents()
    return IndexedDocumentsResponse(count=len(filenames), filenames=filenames)


@router.get(
    "/document/exists",
    response_model=DocumentExistsResponse,
    summary="Check if a document is indexed",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"filename": "amazon-sagemaker-toolkits.md", "exists": True}
                }
            }
        },
        400: {"description": "Missing or blank filename"},
    },
)
async def document_exists(
    filename: str = Query(
        ...,
        min_length=1,
        description="Document filename to check.",
        examples=["amazon-sagemaker-toolkits.md"],
    ),
    search_svc: SearchService = Depends(get_search_service),
) -> DocumentExistsResponse:
    """Check whether a document filename has already been indexed."""

    try:
        exists = await search_svc.document_exists(filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DocumentExistsResponse(filename=filename, exists=exists)


# ------------------------------------------------------------------
# Delete
# ------------------------------------------------------------------


@router.delete(
    "/document/{doc_id}",
    summary="Delete a single document chunk by ID",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"deleted": True}
                }
            }
        },
        404: {"description": "Document not found"},
    },
)
async def delete_document(
    doc_id: str,
    search_svc: SearchService = Depends(get_search_service),
) -> dict[str, bool]:
    """Delete a single document (chunk) by its ID."""

    try:
        deleted = await search_svc.delete_document(doc_id=doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    return {"deleted": True}


@router.delete(
    "/documents",
    response_model=DeleteByFilenameResponse,
    summary="Delete all chunks for a filename",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"filename": "amazon-sagemaker-toolkits.md", "deleted_count": 5}
                }
            }
        },
        400: {"description": "Missing or blank filename"},
    },
)
async def delete_documents_by_filename(
    filename: str = Query(
        ...,
        min_length=1,
        description="Source document filename whose chunks should be deleted.",
        examples=["amazon-sagemaker-toolkits.md"],
    ),
    search_svc: SearchService = Depends(get_search_service),
) -> DeleteByFilenameResponse:
    """Delete all indexed chunks belonging to a given source filename."""

    try:
        count = await search_svc.delete_documents_by_filename(filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DeleteByFilenameResponse(filename=filename, deleted_count=count)
