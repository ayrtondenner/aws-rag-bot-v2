from __future__ import annotations

import logging

from app.models.opensearch import BulkIndexResponse, IndexDocumentRequest
from app.services.document_service import DocumentService
from app.services.opensearch_service import OpenSearchService

logger = logging.getLogger(__name__)


class OpenSearchSetupServiceError(RuntimeError):
    pass


class OpenSearchSetupService:
    def __init__(
        self,
        *,
        opensearch: OpenSearchService,
        document_service: DocumentService,
    ) -> None:
        self._opensearch = opensearch
        self._document_service = document_service

    async def setup_index(self) -> BulkIndexResponse:
        """Read all local sagemaker-docs and bulk-index them into OpenSearch.

        Documents already present in the index are automatically skipped
        (dedup handled by ``OpenSearchService.bulk_index_documents``).

        Returns:
            A ``BulkIndexResponse`` summarising indexed, skipped, and total chunks.
        """
        local = self._document_service.list_local_sagemaker_docs()
        filenames: list[str] = local.get("documents", []) # type: ignore

        docs: list[IndexDocumentRequest] = []
        for fname in filenames:
            try:
                content = self._document_service.get_local_sagemaker_doc_content(
                    filename=fname,
                )
                docs.append(IndexDocumentRequest(filename=fname, content=content))
            except (FileNotFoundError, ValueError):
                logger.warning("Skipping unreadable local doc: %s", fname)
                continue

        if not docs:
            logger.info("OpenSearch setup: no local documents found to index.")
            return BulkIndexResponse(
                total_chunks=0, indexed_count=0, skipped_count=0, results=[],
            )

        result = await self._opensearch.bulk_index_documents(documents=docs)

        logger.info(
            "OpenSearch setup complete: indexed=%d, skipped=%d, total_chunks=%d",
            result.indexed_count,
            result.skipped_count,
            result.total_chunks,
        )

        return result
