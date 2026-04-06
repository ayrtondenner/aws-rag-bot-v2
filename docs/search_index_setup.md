# Search Index Setup

This document describes how to provision and populate the FAISS + BM25 hybrid search infrastructure.

## Architecture

The search system consists of:

1. **AWS Lambda function** (`ragbot-search`) — handles search, indexing, deletion, and stats operations
2. **S3 bucket** — stores serialized index artifacts (`faiss.index`, `corpus.pkl`, `bm25.pkl`)
3. **IAM role** — grants the Lambda function access to S3, Bedrock, and CloudWatch

## Option A: Terraform (recommended)

Deploy all infrastructure with Terraform:

```bash
cd infra
terraform init
terraform apply -var="index_bucket_name=your-bucket-name"
```

This creates:
- S3 bucket with versioning, encryption, and public access block
- IAM execution role with policies for S3, Bedrock, and CloudWatch
- Lambda function (placeholder code — real code is deployed separately)
- CloudWatch log group (14-day retention)

### Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `aws_region` | `us-east-1` | AWS region |
| `index_bucket_name` | (required) | S3 bucket for index artifacts |
| `index_prefix` | `search-index/` | S3 key prefix |
| `lambda_function_name` | `ragbot-search` | Lambda function name |
| `lambda_memory_size` | `512` | Lambda memory (MB) |
| `lambda_timeout` | `30` | Lambda timeout (seconds) |
| `embedding_model_id` | `amazon.titan-embed-text-v2:0` | Bedrock model |
| `embedding_dim` | `1024` | Vector dimensions |
| `bm25_weight` | `0.3` | BM25 fusion weight |
| `vector_weight` | `0.7` | Vector fusion weight |

## Option B: Build script

Build the index and optionally deploy the Lambda in one step:

```bash
# Set required env vars
export SEARCH_INDEX_BUCKET=your-bucket-name
export SEARCH_INDEX_PREFIX=search-index/

# Build index and upload to S3
conda run --prefix .venv python scripts/build_search_index.py

# Also deploy/update the Lambda function
conda run --prefix .venv python scripts/build_search_index.py --deploy-lambda
```

The script:
1. Reads all markdown files from `sagemaker-docs/`
2. Chunks with `RecursiveCharacterTextSplitter` (500 chars, 50 overlap)
3. Generates embeddings via Bedrock Titan V2
4. Builds FAISS (IndexFlatIP) and BM25Okapi indexes
5. Serializes and uploads artifacts to S3
6. Optionally packages `lambda_search/` and creates/updates the Lambda function

## Option C: Automatic at startup

The `SearchSetupService` runs during FastAPI startup and:
1. Checks if the Lambda function exists
2. Checks if FAISS/BM25 artifacts exist in S3
3. Builds and uploads artifacts if missing
4. Creates/updates the Lambda function if missing
5. Bulk-indexes local `sagemaker-docs/` files (idempotent — skips already-indexed docs)

This is **non-fatal** — if setup fails, the app logs a warning and continues. Documents can be indexed later via `POST /search/index-local-docs`.

## Verifying the Setup

After setup, verify via the API:

```bash
# Check index stats
curl http://localhost:8000/search/index/stats

# List indexed documents
curl http://localhost:8000/search/documents

# Test a search
curl -X POST http://localhost:8000/search/search \
  -H "Content-Type: application/json" \
  -d '{"query": "SageMaker training job", "size": 5}'
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SEARCH_LAMBDA_FUNCTION_NAME` | Yes | `ragbot-search` | Lambda function name |
| `SEARCH_INDEX_BUCKET` | Yes | — | S3 bucket for index artifacts |
| `SEARCH_INDEX_PREFIX` | No | `search-index/` | S3 key prefix |
| `BEDROCK_EMBEDDING_MODEL_ID` | No | `amazon.titan-embed-text-v2:0` | Embedding model |
| `BEDROCK_EMBEDDING_DIM` | No | `1024` | Embedding dimensions |
| `BM25_WEIGHT` | No | `0.3` | BM25 fusion weight |
| `VECTOR_WEIGHT` | No | `0.7` | Vector fusion weight |
| `SEARCH_LAMBDA_TIMEOUT_SECONDS` | No | `30` | Lambda invoke timeout |

## Migration from OpenSearch

This project previously used Amazon OpenSearch Serverless for hybrid search. The migration was driven by cost: OpenSearch Serverless costs ~US$536/month idle, while this Lambda-based architecture costs ~US$0.05/month.

See:
- `docs/opensearch-alternatives-comparison.md` — Full comparison report
- `archive/setup_opensearch.py` — Former OpenSearch setup script
- `archive/opensearch_index_setup.md` — Former OpenSearch index setup guide
