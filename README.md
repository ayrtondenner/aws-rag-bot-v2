# aws-rag-bot-v2

RAG (Retrieval-Augmented Generation) backend that ingests documents from AWS S3 and serves a FastAPI API for querying them. The goal is to evolve this into a production-ready, hybrid-search RAG service on AWS (S3 + OpenSearch) with solid testing, docs, and an agent layer.

## Functionality checklist

- [x] S3 integration
- [ ] Hybrid search using OpenSearch
- [ ] FastAPI unit tests
- [ ] Swagger documentation
- [x] Google ADK agent

## Google ADK agent

This repo includes an experimental agent layer built with **Google Agent Development Kit (ADK)**.

### Current capabilities

- **Root agent** that delegates to sub-agents.
- **S3 sub-agent** (`s3_agent`) with tools to:
	- Check whether an S3 bucket exists / is accessible.
	- List objects in the configured bucket (optional prefix).
	- Fetch an object's content by key.
	- Transfer control back to the root agent when a request is not S3-related.

### LLM model

The agent is currently configured to use **AWS Bedrock – Claude Sonnet 4** via LiteLLM.
By default it uses the cross-region inference profile:

- Inference profile ID: `global.anthropic.claude-sonnet-4-20250514-v1:0`
- Model ID: `anthropic.claude-sonnet-4-20250514-v1:0`

### Running the agent (side-by-side with FastAPI)

To avoid port conflicts with the FastAPI app, we run the ADK web UI on **port 8001**:

```bash
adk web --port 8001
```

This lets you keep FastAPI running on its usual port (commonly 8000) while testing the agent in parallel.

## API routes
### S3 API routes

Base path: `/s3`

| Method | Path | Description | Query params |
| --- | --- | --- | --- |
| GET | `/s3/bucket/exists` | Check whether the configured S3 bucket exists (and is accessible). | _None_ |
| GET | `/s3/bucket/files/count` | Count objects in the configured S3 bucket. | `prefix` (optional) |
| GET | `/s3/files` | List objects in the configured S3 bucket. | `prefix` (optional) |
| GET | `/s3/file/content` | Get raw content of an object by key. | `file_name` (required) |

### Document API routes

Base path: `/document`

| Method | Path | Description | Query params | Body |
| --- | --- | --- | --- | --- |
| POST | `/document/chunks` | Split text into overlapping chunks. | `chunk_size` (optional), `chunk_overlap` (optional) | `{ "text": "..." }` |
| POST | `/document/embed` | Generate an embedding vector for input text (Amazon Bedrock). | _None_ | `{ "text": "..." }` |
