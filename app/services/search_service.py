from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

import boto3
from tqdm import tqdm

from app.models.search import (
    BulkIndexResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    IndexStatsResponse,
    SearchHit,
    SearchResponse,
)
from app.services.config import SearchConfig
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


class SearchServiceError(RuntimeError):
    """Raised when a search backend operation fails."""


class SearchService:
    """Service for indexing and searching documents via Lambda (FAISS + BM25 + S3).

    Uses ``boto3.client("lambda").invoke()`` wrapped in
    ``asyncio.to_thread()`` so FastAPI routes remain non-blocking.
    """

    def __init__(self, config: SearchConfig, document_service: DocumentService) -> None:
        self._config = config
        self._document_service = document_service

    # ------------------------------------------------------------------
    # Client factory (replaceable in tests)
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        """Build a boto3 Lambda client."""

        return boto3.client("lambda", region_name=self._config.region)

    # ------------------------------------------------------------------
    # Lambda invocation
    # ------------------------------------------------------------------

    def _invoke_lambda(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke the search Lambda function synchronously.

        Args:
            payload: JSON-serializable dict to send as the event.

        Returns:
            Parsed JSON response from Lambda.
        """

        client = self._client()
        response = client.invoke(
            FunctionName=self._config.lambda_function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )
        response_payload = json.loads(response["Payload"].read())

        if response.get("FunctionError"):
            error_msg = response_payload.get("errorMessage", "Unknown Lambda error")
            raise SearchServiceError(f"Lambda execution failed: {error_msg}")

        if "error" in response_payload:
            raise SearchServiceError(f"Search operation failed: {response_payload['error']}")

        return response_payload

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

        try:
            raw = await asyncio.to_thread(
                self._invoke_lambda, {"action": "document_exists", "filename": filename}
            )
            return raw.get("exists", False)
        except SearchServiceError:
            raise
        except Exception as exc:
            logger.exception("Search document_exists failed for filename=%s", filename)
            raise SearchServiceError(f"Failed to check document existence: {filename}") from exc

    async def list_indexed_documents(self) -> list[str]:
        """Return a deduplicated, sorted list of all indexed ``filename`` values.

        Returns:
            Sorted list of unique filenames currently in the index.
        """

        try:
            raw = await asyncio.to_thread(
                self._invoke_lambda, {"action": "list_documents"}
            )
            return raw.get("filenames", [])
        except SearchServiceError:
            raise
        except Exception as exc:
            logger.exception("Search list_indexed_documents failed")
            raise SearchServiceError("Failed to list indexed documents") from exc

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

        payload = {
            "action": "index_documents",
            "documents": [{"filename": filename, "chunks": chunks}],
        }

        try:
            raw = await asyncio.to_thread(self._invoke_lambda, payload)
        except SearchServiceError:
            raise
        except Exception as exc:
            logger.exception("Search index_document Lambda call failed for %s", filename)
            raise SearchServiceError(f"Failed to index document: {filename}") from exc

        doc_ids = raw.get("doc_ids", [])
        return IndexDocumentResponse(
            filename=filename, chunk_count=len(doc_ids), doc_ids=doc_ids, skipped=False,
        )

    async def bulk_index_documents(
        self,
        *,
        documents: list[IndexDocumentRequest],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_concurrency: int = 5,
    ) -> BulkIndexResponse:
        """Bulk-index multiple documents in parallel, skipping those already indexed.

        Uses ``asyncio.gather()`` with a semaphore to process up to
        *max_concurrency* documents at a time.

        Args:
            documents: List of documents to index.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.
            max_concurrency: Maximum number of documents indexed in parallel.

        Returns:
            A ``BulkIndexResponse`` summarising the operation.
        """

        sem = asyncio.Semaphore(max_concurrency)

        async def _index_one(doc: IndexDocumentRequest) -> IndexDocumentResponse:
            async with sem:
                resp = await self.index_document(
                    filename=doc.filename,
                    content=doc.content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            pbar.update(1)
            return resp

        with tqdm(total=len(documents), desc="Indexing documents", unit="doc") as pbar:
            results = list(await asyncio.gather(*[_index_one(doc) for doc in documents]))

        total_chunks = sum(r.chunk_count for r in results)
        indexed_count = sum(1 for r in results if not r.skipped)
        skipped_count = sum(1 for r in results if r.skipped)

        return BulkIndexResponse(
            total_chunks=total_chunks,
            indexed_count=indexed_count,
            skipped_count=skipped_count,
            results=results,
        )

    # ------------------------------------------------------------------
    # Build index (bulk from raw documents)
    # ------------------------------------------------------------------

    async def build_index(
        self,
        *,
        documents: list[dict[str, str]],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        skip_existing: bool = True,
        batch_size: int = 50,
    ) -> dict[str, int]:
        """Build the search index from raw documents via Lambda.

        Sends documents in batches to the Lambda ``build_index`` action,
        which handles chunking, embedding, and index persistence.

        Args:
            documents: List of ``{filename, content}`` dicts.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.
            skip_existing: Skip documents already in the index.
            batch_size: Number of documents per Lambda invocation.

        Returns:
            Aggregated counts: ``{processed_count, skipped_count, total_chunks_added}``.
        """

        totals = {"processed_count": 0, "skipped_count": 0, "total_chunks_added": 0}

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            payload = {
                "action": "build_index",
                "documents": batch,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "skip_existing": skip_existing,
            }

            try:
                raw = await asyncio.to_thread(self._invoke_lambda, payload)
            except SearchServiceError:
                raise
            except Exception as exc:
                logger.exception("build_index Lambda call failed for batch %d", i)
                raise SearchServiceError(f"Failed to build index (batch starting at {i})") from exc

            totals["processed_count"] += raw.get("processed_count", 0)
            totals["skipped_count"] += raw.get("skipped_count", 0)
            totals["total_chunks_added"] += raw.get("total_chunks_added", 0)

        return totals

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

        payload = {
            "action": "search",
            "query": query,
            "size": size,
            "search_type": search_type,
        }

        try:
            raw = await asyncio.to_thread(self._invoke_lambda, payload)
        except SearchServiceError:
            raise
        except Exception as exc:
            logger.exception("Search Lambda invocation failed")
            raise SearchServiceError("Failed to execute search") from exc

        hits = [SearchHit(**h) for h in raw.get("hits", [])]
        return SearchResponse(
            query=query,
            search_type=search_type,
            total_hits=raw.get("total_hits", len(hits)),
            hits=hits,
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_document(self, *, doc_id: str) -> bool:
        """Delete a single document (chunk) by its ID.

        Args:
            doc_id: The document chunk ID.

        Returns:
            ``True`` if the document was deleted, ``False`` if it was not found.
        """

        if not doc_id or not doc_id.strip():
            raise ValueError("doc_id must be provided")

        try:
            raw = await asyncio.to_thread(
                self._invoke_lambda, {"action": "delete_document", "doc_id": doc_id}
            )
            return raw.get("deleted", False)
        except SearchServiceError:
            raise
        except Exception as exc:
            logger.exception("Search delete_document failed for doc_id=%s", doc_id)
            raise SearchServiceError(f"Failed to delete document: {doc_id}") from exc

    async def delete_documents_by_filename(self, *, filename: str) -> int:
        """Delete all chunks for a given source *filename*.

        Args:
            filename: The source document filename whose chunks should be removed.

        Returns:
            Number of chunks deleted.
        """

        if not filename or not filename.strip():
            raise ValueError("filename must be provided")

        try:
            raw = await asyncio.to_thread(
                self._invoke_lambda, {"action": "delete_by_filename", "filename": filename}
            )
            return raw.get("deleted_count", 0)
        except SearchServiceError:
            raise
        except Exception as exc:
            logger.exception("Search delete_documents_by_filename failed for filename=%s", filename)
            raise SearchServiceError(f"Failed to delete documents for filename: {filename}") from exc

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_index_stats(self) -> IndexStatsResponse:
        """Return basic index statistics.

        Returns:
            An ``IndexStatsResponse`` with document count and index status.
        """

        try:
            raw = await asyncio.to_thread(self._invoke_lambda, {"action": "get_stats"})
            return IndexStatsResponse(
                index_name=raw.get("index_name", "sagemaker-docs"),
                doc_count=raw.get("doc_count", 0),
                status=raw.get("status", "available"),
            )
        except SearchServiceError:
            raise
        except Exception as exc:
            logger.exception("Search get_index_stats failed")
            raise SearchServiceError("Failed to get index stats") from exc
