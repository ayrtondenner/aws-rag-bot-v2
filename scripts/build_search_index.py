"""Build FAISS + BM25 search index artifacts and upload to S3.

Reads all documents from ``sagemaker-docs/``, chunks them, generates
embeddings via Bedrock Titan, builds FAISS and BM25 indexes, and uploads
the serialized artifacts to S3.

Usage::

    conda run --prefix .venv python scripts/build_search_index.py

    # Also deploy/update the Lambda function:
    conda run --prefix .venv python scripts/build_search_index.py --deploy-lambda

Prerequisites:
    - AWS credentials configured
    - S3 bucket must exist (created by app startup or Terraform)
    - Environment variables: SEARCH_INDEX_BUCKET, SEARCH_INDEX_PREFIX (optional)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import pickle
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

import boto3
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Force unbuffered output so progress is visible in all terminals
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

S3_BUCKET = os.getenv("SEARCH_INDEX_BUCKET", "")
S3_PREFIX = os.getenv("SEARCH_INDEX_PREFIX", "search-index/")
EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBEDDING_DIM = int(os.getenv("BEDROCK_EMBEDDING_DIM", "1024"))
LAMBDA_FUNCTION_NAME = os.getenv("SEARCH_LAMBDA_FUNCTION_NAME", "ragbot-search")
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

DOCS_DIR = _PROJECT_ROOT / "sagemaker-docs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_all_docs() -> list[dict[str, str]]:
    """Read all markdown files from the sagemaker-docs directory."""

    docs = []
    if not DOCS_DIR.exists():
        logger.warning("Docs directory not found: %s", DOCS_DIR)
        return docs

    for path in sorted(DOCS_DIR.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
            if content.strip():
                docs.append({"filename": path.name, "content": content})
        except Exception:
            logger.warning("Failed to read %s, skipping", path.name)
    return docs


def _chunk_documents(
    docs: list[dict[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict[str, str]]:
    """Split documents into chunks, returning metadata per chunk."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    corpus: list[dict[str, str]] = []
    for doc in docs:
        chunks = splitter.split_text(doc["content"])
        for chunk in chunks:
            corpus.append({
                "doc_id": str(uuid.uuid4()),
                "filename": doc["filename"],
                "content": chunk,
            })
    return corpus


def _embed_text(bedrock_client: object, text: str) -> np.ndarray:
    """Generate embedding via Bedrock."""

    response = bedrock_client.invoke_model(  # type: ignore[union-attr]
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
    )
    body = json.loads(response["body"].read())
    vec = np.array(body["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _embed_corpus(corpus: list[dict[str, str]]) -> np.ndarray:
    """Embed all corpus chunks, returning a (N, dim) array."""

    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    total = len(corpus)
    vectors = []
    for i, doc in enumerate(tqdm(corpus, desc="Generating embeddings", unit="chunk", file=sys.stdout)):
        vec = _embed_text(bedrock, doc["content"])
        vectors.append(vec)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Embedded {i + 1}/{total} chunks", flush=True)
    return np.vstack(vectors)


def _build_indexes(
    corpus: list[dict[str, str]],
    vectors: np.ndarray,
) -> tuple[faiss.Index, BM25Okapi]:
    """Build FAISS (IndexFlatIP) and BM25Okapi indexes."""

    # FAISS — inner product on L2-normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    # BM25
    tokenized = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    return index, bm25


def _upload_artifacts(
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    corpus: list[dict[str, str]],
) -> None:
    """Serialize and upload artifacts to S3."""

    s3 = boto3.client("s3", region_name=AWS_REGION)
    prefix = S3_PREFIX

    # FAISS index
    tmp = tempfile.gettempdir()
    faiss_path = os.path.join(tmp, "faiss.index")
    faiss.write_index(faiss_index, faiss_path)
    s3.upload_file(faiss_path, S3_BUCKET, f"{prefix}faiss.index")
    logger.info("Uploaded faiss.index (%d vectors)", faiss_index.ntotal)

    # Corpus metadata
    corpus_bytes = pickle.dumps(corpus)
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}corpus.pkl", Body=corpus_bytes)
    logger.info("Uploaded corpus.pkl (%d documents)", len(corpus))

    # BM25 index
    bm25_bytes = pickle.dumps(bm25_index)
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}bm25.pkl", Body=bm25_bytes)
    logger.info("Uploaded bm25.pkl")


def _deploy_lambda() -> None:
    """Package lambda_search/ and create/update the Lambda function."""

    lambda_dir = _PROJECT_ROOT / "lambda_search"
    if not lambda_dir.exists():
        logger.error("lambda_search/ directory not found")
        return

    # Create zip archive of lambda_search/
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(lambda_dir.rglob("*.py")):
            arcname = path.relative_to(lambda_dir)
            zf.write(path, arcname)
    zip_bytes = buf.getvalue()

    client = boto3.client("lambda", region_name=AWS_REGION)

    try:
        client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
        # Update existing
        client.update_function_code(
            FunctionName=LAMBDA_FUNCTION_NAME,
            ZipFile=zip_bytes,
        )
        logger.info("Updated Lambda function: %s", LAMBDA_FUNCTION_NAME)
    except client.exceptions.ResourceNotFoundException:
        # Need IAM role ARN — try to find it
        iam = boto3.client("iam")
        role_name = f"{LAMBDA_FUNCTION_NAME}-execution-role"
        try:
            role = iam.get_role(RoleName=role_name)
            role_arn = role["Role"]["Arn"]
        except iam.exceptions.NoSuchEntityException:
            logger.error(
                "Lambda function and IAM role '%s' do not exist. "
                "Create them via Terraform first, then re-run with --deploy-lambda.",
                role_name,
            )
            return

        client.create_function(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Runtime="python3.11",
            Role=role_arn,
            Handler="handler.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Timeout=30,
            MemorySize=512,
            Environment={
                "Variables": {
                    "SEARCH_INDEX_BUCKET": S3_BUCKET,
                    "SEARCH_INDEX_PREFIX": S3_PREFIX,
                    "BEDROCK_EMBEDDING_MODEL_ID": EMBEDDING_MODEL_ID,
                    "BEDROCK_EMBEDDING_DIM": str(EMBEDDING_DIM),
                    "BM25_WEIGHT": os.getenv("BM25_WEIGHT", "0.3"),
                    "VECTOR_WEIGHT": os.getenv("VECTOR_WEIGHT", "0.7"),
                },
            },
        )
        logger.info("Created Lambda function: %s", LAMBDA_FUNCTION_NAME)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and upload FAISS+BM25 search index")
    parser.add_argument(
        "--deploy-lambda",
        action="store_true",
        help="Also create/update the Lambda function",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    if not S3_BUCKET:
        logger.error("SEARCH_INDEX_BUCKET env var is required")
        sys.exit(1)

    # 1. Read documents
    logger.info("Reading documents from %s", DOCS_DIR)
    docs = _read_all_docs()
    if not docs:
        logger.error("No documents found")
        sys.exit(1)
    logger.info("Found %d documents", len(docs))

    # 2. Chunk
    logger.info("Chunking documents (size=%d, overlap=%d)", args.chunk_size, args.chunk_overlap)
    corpus = _chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    logger.info("Created %d chunks", len(corpus))

    # 3. Embed
    logger.info("Generating embeddings via Bedrock (%s)", EMBEDDING_MODEL_ID)
    vectors = _embed_corpus(corpus)
    logger.info("Generated %d embeddings of dimension %d", len(vectors), vectors.shape[1])

    # 4. Build indexes
    logger.info("Building FAISS + BM25 indexes")
    faiss_index, bm25_index = _build_indexes(corpus, vectors)

    # 5. Upload to S3
    logger.info("Uploading artifacts to s3://%s/%s", S3_BUCKET, S3_PREFIX)
    _upload_artifacts(faiss_index, bm25_index, corpus)

    # 6. Optionally deploy Lambda
    if args.deploy_lambda:
        logger.info("Deploying Lambda function")
        _deploy_lambda()

    logger.info("Done! Index with %d chunks uploaded to S3.", len(corpus))


if __name__ == "__main__":
    main()
