from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IndexDocumentRequest(BaseModel):
    """Request body for indexing a single document into OpenSearch."""

    filename: str = Field(..., description="Source document filename (e.g. 'amazon-sagemaker-toolkits.md').")
    content: str = Field(..., description="Full text content of the document.")


class IndexDocumentResponse(BaseModel):
    """Response after indexing (or skipping) a single document."""

    filename: str
    chunk_count: int
    doc_ids: list[str]
    skipped: bool = False


class BulkIndexRequest(BaseModel):
    """Request body for bulk-indexing multiple documents."""

    documents: list[IndexDocumentRequest] = Field(..., min_length=1, description="Documents to index.")
    chunk_size: int = Field(default=500, ge=1, description="Target chunk size in characters.")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap size in characters between chunks.")
    max_concurrency: int = Field(default=5, ge=1, le=20, description="Maximum documents indexed in parallel.")


class BulkIndexResponse(BaseModel):
    """Response after bulk-indexing multiple documents."""

    total_chunks: int
    indexed_count: int
    skipped_count: int
    results: list[IndexDocumentResponse]


class DocumentExistsResponse(BaseModel):
    """Response for checking whether a document filename is already indexed."""

    filename: str
    exists: bool


class IndexedDocumentsResponse(BaseModel):
    """Response listing all unique indexed document filenames."""

    count: int
    filenames: list[str]


class SearchRequest(BaseModel):
    """Request body for searching OpenSearch."""

    query: str = Field(..., min_length=1, description="Search query text.")
    size: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return.")
    search_type: Literal["hybrid", "text", "vector"] = Field(
        default="hybrid",
        description="Search strategy: 'hybrid' (BM25 + neural), 'text' (BM25 only), or 'vector' (neural only).",
    )


class SearchHit(BaseModel):
    """A single search result hit."""

    doc_id: str
    score: float
    filename: str
    content: str


class SearchResponse(BaseModel):
    """Response from a search query."""

    query: str
    search_type: str
    total_hits: int
    hits: list[SearchHit]


class IndexStatsResponse(BaseModel):
    """Response with index-level statistics."""

    index_name: str
    doc_count: int
    status: str


class DeleteByFilenameResponse(BaseModel):
    """Response after deleting all chunks belonging to a given filename."""

    filename: str
    deleted_count: int
