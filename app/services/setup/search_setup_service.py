from __future__ import annotations

import asyncio
import io
import logging
import os
import zipfile
from pathlib import Path

import boto3

from app.models.search import BulkIndexResponse, IndexDocumentRequest
from app.services.config import SearchConfig
from app.services.document_service import DocumentService
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LAMBDA_DIR = _PROJECT_ROOT / "lambda_search"

EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBEDDING_DIM = int(os.getenv("BEDROCK_EMBEDDING_DIM", "1024"))


class SearchSetupServiceError(RuntimeError):
    pass


class SearchSetupService:
    """Startup service that ensures Lambda + S3 search artifacts exist.

    Checks at app startup whether the Lambda function and FAISS/BM25
    index artifacts exist. If missing, builds and deploys them.
    """

    def __init__(
        self,
        *,
        search: SearchService,
        document_service: DocumentService,
        config: SearchConfig,
    ) -> None:
        self._search = search
        self._document_service = document_service
        self._config = config

    async def setup_index(self) -> BulkIndexResponse:
        """Ensure Lambda + S3 artifacts exist; build and deploy if missing.

        Returns:
            BulkIndexResponse summarising what was indexed.
        """

        lambda_exists = await asyncio.to_thread(self._lambda_exists)
        artifacts_exist = await asyncio.to_thread(self._artifacts_exist)

        if lambda_exists and artifacts_exist:
            logger.info("Search setup: Lambda and artifacts already exist.")
            return await self._index_local_docs()

        if not artifacts_exist:
            logger.info("Search setup: Building FAISS/BM25 artifacts...")
            await self._build_and_upload_artifacts()

        if not lambda_exists:
            logger.info("Search setup: Creating Lambda function...")
            await asyncio.to_thread(self._create_or_update_lambda)

        return await self._index_local_docs()

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    def _lambda_exists(self) -> bool:
        client = boto3.client("lambda", region_name=self._config.region)
        try:
            client.get_function(FunctionName=self._config.lambda_function_name)
            return True
        except client.exceptions.ResourceNotFoundException:
            return False
        except Exception:
            logger.warning("Could not check Lambda existence", exc_info=True)
            return False

    def _artifacts_exist(self) -> bool:
        s3 = boto3.client("s3", region_name=self._config.region)
        prefix = self._config.index_prefix
        try:
            s3.head_object(Bucket=self._config.index_bucket, Key=f"{prefix}faiss.index")
            s3.head_object(Bucket=self._config.index_bucket, Key=f"{prefix}corpus.pkl")
            s3.head_object(Bucket=self._config.index_bucket, Key=f"{prefix}bm25.pkl")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Artifact building
    # ------------------------------------------------------------------

    async def _build_and_upload_artifacts(self) -> None:
        """Read local docs and delegate index building to Lambda."""

        local = self._document_service.list_local_sagemaker_docs()
        filenames: list[str] = local.get("documents", [])  # type: ignore

        documents: list[dict[str, str]] = []
        for fname in filenames:
            try:
                content = self._document_service.get_local_sagemaker_doc_content(filename=fname)
                documents.append({"filename": fname, "content": content})
            except (FileNotFoundError, ValueError):
                logger.warning("Skipping unreadable doc: %s", fname)

        if not documents:
            logger.warning("No documents found to build index")
            return

        logger.info("Building index via Lambda for %d documents...", len(documents))
        result = await self._search.build_index(documents=documents)
        logger.info(
            "Index built via Lambda: processed=%d, skipped=%d, chunks=%d",
            result["processed_count"],
            result["skipped_count"],
            result["total_chunks_added"],
        )

    # ------------------------------------------------------------------
    # Lambda deployment
    # ------------------------------------------------------------------

    def _create_or_update_lambda(self) -> None:
        """Package lambda_search/ as zip and create/update the Lambda function."""

        if not _LAMBDA_DIR.exists():
            logger.error("lambda_search/ directory not found at %s", _LAMBDA_DIR)
            return

        # Create zip archive
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(_LAMBDA_DIR.rglob("*.py")):
                arcname = path.relative_to(_LAMBDA_DIR)
                zf.write(path, arcname)
        zip_bytes = buf.getvalue()

        client = boto3.client("lambda", region_name=self._config.region)

        try:
            client.get_function(FunctionName=self._config.lambda_function_name)
            client.update_function_code(
                FunctionName=self._config.lambda_function_name,
                ZipFile=zip_bytes,
            )
            logger.info("Updated Lambda function: %s", self._config.lambda_function_name)
        except client.exceptions.ResourceNotFoundException:
            # Look for IAM role
            iam = boto3.client("iam")
            role_name = f"{self._config.lambda_function_name}-execution-role"
            try:
                role = iam.get_role(RoleName=role_name)
                role_arn = role["Role"]["Arn"]
            except iam.exceptions.NoSuchEntityException:
                logger.error(
                    "IAM role '%s' not found. Create it via Terraform first.",
                    role_name,
                )
                return

            client.create_function(
                FunctionName=self._config.lambda_function_name,
                Runtime="python3.11",
                Role=role_arn,
                Handler="handler.lambda_handler",
                Code={"ZipFile": zip_bytes},
                Timeout=self._config.lambda_timeout_seconds,
                MemorySize=512,
                Environment={
                    "Variables": {
                        "SEARCH_INDEX_BUCKET": self._config.index_bucket,
                        "SEARCH_INDEX_PREFIX": self._config.index_prefix,
                        "BEDROCK_EMBEDDING_MODEL_ID": EMBEDDING_MODEL_ID,
                        "BEDROCK_EMBEDDING_DIM": str(EMBEDDING_DIM),
                        "BM25_WEIGHT": str(self._config.bm25_weight),
                        "VECTOR_WEIGHT": str(self._config.vector_weight),
                    },
                },
            )
            logger.info("Created Lambda function: %s", self._config.lambda_function_name)

    # ------------------------------------------------------------------
    # Local docs indexing
    # ------------------------------------------------------------------

    async def _index_local_docs(self) -> BulkIndexResponse:
        """Delegate to SearchService.bulk_index_documents for any new local docs."""

        local = self._document_service.list_local_sagemaker_docs()
        filenames: list[str] = local.get("documents", [])  # type: ignore

        docs: list[IndexDocumentRequest] = []
        for fname in filenames:
            try:
                content = self._document_service.get_local_sagemaker_doc_content(filename=fname)
                docs.append(IndexDocumentRequest(filename=fname, content=content))
            except (FileNotFoundError, ValueError):
                logger.warning("Skipping unreadable local doc: %s", fname)
                continue

        if not docs:
            logger.info("Search setup: no local documents found to index.")
            return BulkIndexResponse(
                total_chunks=0, indexed_count=0, skipped_count=0, results=[],
            )

        result = await self._search.bulk_index_documents(documents=docs)

        logger.info(
            "Search setup complete: indexed=%d, skipped=%d, total_chunks=%d",
            result.indexed_count,
            result.skipped_count,
            result.total_chunks,
        )

        return result
