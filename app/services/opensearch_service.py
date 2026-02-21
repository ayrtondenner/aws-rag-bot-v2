from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from app.models.opensearch import (
    BulkIndexResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    IndexStatsResponse,
    SearchHit,
    SearchResponse,
)
from app.services.config import OpenSearchConfig
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

# Pipeline names — must match what the dashboard setup script creates.
INGEST_PIPELINE = "sagemaker-docs-ingest-pipeline"
SEARCH_PIPELINE = "sagemaker-docs-search-pipeline"

# Field that holds the embedding vector (auto-populated by the ingest pipeline).
EMBEDDING_FIELD = "content_embedding"


class OpenSearchServiceError(RuntimeError):
    """Raised when an OpenSearch data-plane operation fails."""


class OpenSearchService:
    """Service for indexing and searching documents in OpenSearch Serverless.

    Uses the sync ``opensearch-py`` client with SigV4 auth, wrapped in
    ``asyncio.to_thread()`` so FastAPI routes remain non-blocking.
    """

    def __init__(self, config: OpenSearchConfig, document_service: DocumentService) -> None:
        self._config = config
        self._document_service = document_service

    # ------------------------------------------------------------------
    # Client factory (replaceable in tests)
    # ------------------------------------------------------------------

    def _client(self) -> OpenSearch:
        """Build a new ``OpenSearch`` client with SigV4 auth for AOSS.

        Returns:
            A configured OpenSearch client instance.
        """

        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise OpenSearchServiceError("AWS credentials not available")

        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self._config.region or "us-east-1",
            self._config.service_name,
            session_token=credentials.token,
        )

        # Strip the scheme for the ``hosts`` list — opensearch-py wants just the host.
        host = self._config.endpoint.replace("https://", "").replace("http://", "").rstrip("/")

        return OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=self._config.timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Document existence helpers
    # ------------------------------------------------------------------

    async def document_exists(self, *, filename: str) -> bool:
        """Check whether any chunks for *filename* are already indexed.

        Args:
            filename: The source document filename to look up.

        Returns:
            ``True`` if at least one chunk exists for this filename.
        """

        if not filename or not filename.strip():
            raise ValueError("filename must be provided")

        def _query() -> bool:
            client = self._client()
            resp = client.search(
                index=self._config.index_name,
                body={
                    "query": {"term": {"filename": filename}},
                    "size": 0,
                    "track_total_hits": True,
                },
            )
            total = resp.get("hits", {}).get("total", {}).get("value", 0)
            return total > 0

        try:
            return await asyncio.to_thread(_query)
        except Exception as exc:
            logger.exception("OpenSearch document_exists failed for filename=%s", filename)
            raise OpenSearchServiceError(f"Failed to check document existence: {filename}") from exc

    async def list_indexed_documents(self) -> list[str]:
        """Return a deduplicated, sorted list of all indexed ``filename`` values.

        Returns:
            Sorted list of unique filenames currently in the index.
        """

        def _query() -> list[str]:
            client = self._client()
            resp = client.search(
                index=self._config.index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "unique_filenames": {
                            "terms": {"field": "filename", "size": 10000},
                        }
                    },
                },
            )
            buckets = resp.get("aggregations", {}).get("unique_filenames", {}).get("buckets", [])
            return sorted(b["key"] for b in buckets)

        try:
            return await asyncio.to_thread(_query)
        except Exception as exc:
            logger.exception("OpenSearch list_indexed_documents failed")
            raise OpenSearchServiceError("Failed to list indexed documents") from exc

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_document(
        self,
        *,
        filename: str,
        content: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> IndexDocumentResponse:
        """Index a single document, splitting it into chunks first.

        Skips indexing if the filename is already present in the index.

        Args:
            filename: Source document filename.
            content: Full text content of the document.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            An ``IndexDocumentResponse`` indicating what happened.
        """

        if not filename or not filename.strip():
            raise ValueError("filename must be provided")
        if not content or not content.strip():
            raise ValueError("content must be provided")

        # Skip if already indexed.
        if await self.document_exists(filename=filename):
            return IndexDocumentResponse(filename=filename, chunk_count=0, doc_ids=[], skipped=True)

        chunks = self._document_service.chunk_text(
            text=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            return IndexDocumentResponse(filename=filename, chunk_count=0, doc_ids=[], skipped=False)

        doc_ids = await self._bulk_index_chunks(filename=filename, chunks=chunks)
        return IndexDocumentResponse(filename=filename, chunk_count=len(doc_ids), doc_ids=doc_ids, skipped=False)

    # TODO: Check if this can happens in parallel, maybe using asyncio or similar
    async def bulk_index_documents(
        self,
        *,
        documents: list[IndexDocumentRequest],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> BulkIndexResponse:
        """Bulk-index multiple documents, skipping those already indexed.

        Args:
            documents: List of documents to index.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            A ``BulkIndexResponse`` summarising the operation.
        """

        results: list[IndexDocumentResponse] = []
        total_chunks = 0
        indexed_count = 0
        skipped_count = 0

        for doc in documents:
            resp = await self.index_document(
                filename=doc.filename,
                content=doc.content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            results.append(resp)
            total_chunks += resp.chunk_count
            if resp.skipped:
                skipped_count += 1
            else:
                indexed_count += 1

        return BulkIndexResponse(
            total_chunks=total_chunks,
            indexed_count=indexed_count,
            skipped_count=skipped_count,
            results=results,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        *,
        query: str,
        size: int = 10,
        search_type: Literal["hybrid", "text", "vector"] = "hybrid",
    ) -> SearchResponse:
        """Search the index using the specified strategy.

        Args:
            query: The user's search query.
            size: Maximum number of results.
            search_type: One of ``"hybrid"``, ``"text"``, or ``"vector"``.

        Returns:
            A ``SearchResponse`` containing matching hits.
        """

        if not query or not query.strip():
            raise ValueError("query must be provided")

        body = self._build_search_body(query=query, size=size, search_type=search_type)
        params = self._build_search_params(search_type=search_type)

        def _query() -> dict[str, Any]:
            client = self._client()
            return client.search(index=self._config.index_name, body=body, params=params)

        try:
            raw = await asyncio.to_thread(_query)
        except Exception as exc:
            logger.exception("OpenSearch search failed")
            raise OpenSearchServiceError("Failed to execute search") from exc

        hits: list[SearchHit] = []
        for h in raw.get("hits", {}).get("hits", []):
            source = h.get("_source", {})
            hits.append(
                SearchHit(
                    doc_id=h.get("_id", ""),
                    score=h.get("_score", 0.0) or 0.0,
                    filename=source.get("filename", ""),
                    content=source.get("content", ""),
                )
            )

        total_hits = raw.get("hits", {}).get("total", {}).get("value", len(hits))

        return SearchResponse(
            query=query,
            search_type=search_type,
            total_hits=total_hits,
            hits=hits,
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_document(self, *, doc_id: str) -> bool:
        """Delete a single document (chunk) by its OpenSearch ``_id``.

        Args:
            doc_id: The OpenSearch document ID.

        Returns:
            ``True`` if the document was deleted, ``False`` if it was not found.
        """

        if not doc_id or not doc_id.strip():
            raise ValueError("doc_id must be provided")

        def _delete() -> bool:
            client = self._client()
            resp = client.delete(index=self._config.index_name, id=doc_id)
            return resp.get("result") == "deleted"

        try:
            return await asyncio.to_thread(_delete)
        except Exception as exc:
            error_str = str(exc)
            if "NotFoundError" in type(exc).__name__ or "404" in error_str:
                return False
            logger.exception("OpenSearch delete_document failed for doc_id=%s", doc_id)
            raise OpenSearchServiceError(f"Failed to delete document: {doc_id}") from exc

    async def delete_documents_by_filename(self, *, filename: str) -> int:
        """Delete all chunks for a given source *filename*.

        Args:
            filename: The source document filename whose chunks should be removed.

        Returns:
            Number of chunks deleted.
        """

        if not filename or not filename.strip():
            raise ValueError("filename must be provided")

        def _delete() -> int:
            client = self._client()
            resp = client.delete_by_query(
                index=self._config.index_name,
                body={"query": {"term": {"filename": filename}}},
            )
            return resp.get("deleted", 0)

        try:
            return await asyncio.to_thread(_delete)
        except Exception as exc:
            logger.exception("OpenSearch delete_documents_by_filename failed for filename=%s", filename)
            raise OpenSearchServiceError(f"Failed to delete documents for filename: {filename}") from exc

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_index_stats(self) -> IndexStatsResponse:
        """Return basic index statistics.

        Returns:
            An ``IndexStatsResponse`` with document count and index status.
        """

        def _stats() -> IndexStatsResponse:
            client = self._client()
            resp = client.count(index=self._config.index_name)
            doc_count = resp.get("count", 0)
            return IndexStatsResponse(
                index_name=self._config.index_name,
                doc_count=doc_count,
                status="available",
            )

        try:
            return await asyncio.to_thread(_stats)
        except Exception as exc:
            logger.exception("OpenSearch get_index_stats failed")
            raise OpenSearchServiceError("Failed to get index stats") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _bulk_index_chunks(self, *, filename: str, chunks: list[str]) -> list[str]:
        """Index a list of text chunks via the OpenSearch ``_bulk`` API.

        Args:
            filename: Source document filename (stored on every chunk doc).
            chunks: List of text chunks to index.

        Returns:
            List of assigned OpenSearch document IDs.
        """

        actions: list[dict[str, Any]] = []
        for chunk in chunks:
            actions.append({"index": {"_index": self._config.index_name}})
            actions.append({"filename": filename, "content": chunk})

        def _bulk() -> list[str]:
            client = self._client()
            resp = client.bulk(body=actions)
            if resp.get("errors"):
                failed = [
                    item for item in resp.get("items", [])
                    if item.get("index", {}).get("error")
                ]
                logger.error("Bulk index had %d errors: %s", len(failed), failed[:3])
                raise OpenSearchServiceError(f"Bulk indexing had {len(failed)} error(s)")
            return [item["index"]["_id"] for item in resp.get("items", [])]

        try:
            return await asyncio.to_thread(_bulk)
        except OpenSearchServiceError:
            raise
        except Exception as exc:
            logger.exception("OpenSearch bulk index failed for filename=%s", filename)
            raise OpenSearchServiceError(f"Failed to bulk-index chunks for: {filename}") from exc

    def _build_search_body(
        self,
        *,
        query: str,
        size: int,
        search_type: Literal["hybrid", "text", "vector"],
    ) -> dict[str, Any]:
        """Build the OpenSearch query body for the given *search_type*.

        Args:
            query: User's search text.
            size: Maximum hits.
            search_type: Search strategy.

        Returns:
            A dict ready to be passed as ``body`` to ``client.search()``.
        """

        if search_type == "text":
            return {
                "size": size,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["filename", "content"],
                    }
                },
            }

        if search_type == "vector":
            return {
                "size": size,
                "query": {
                    "neural": {
                        EMBEDDING_FIELD: {
                            "query_text": query,
                            "k": size,
                        }
                    }
                },
            }

        # hybrid — combine BM25 + neural via the search pipeline
        return {
            "size": size,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["filename", "content"],
                            }
                        },
                        {
                            "neural": {
                                EMBEDDING_FIELD: {
                                    "query_text": query,
                                    "k": size,
                                }
                            }
                        },
                    ]
                }
            },
        }

    @staticmethod
    def _build_search_params(*, search_type: Literal["hybrid", "text", "vector"]) -> dict[str, str]:
        """Return extra request params (e.g. ``search_pipeline``) for the search type.

        Args:
            search_type: Search strategy.

        Returns:
            Dict of query-string parameters for the OpenSearch request.
        """

        if search_type == "hybrid":
            return {"search_pipeline": SEARCH_PIPELINE}
        return {}
