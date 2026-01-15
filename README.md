# aws-rag-bot-v2

RAG (Retrieval-Augmented Generation) backend that ingests documents from AWS S3 and serves a FastAPI API for querying them. The goal is to evolve this into a production-ready, hybrid-search RAG service on AWS (S3 + OpenSearch) with solid testing, docs, and an agent layer.

## Functionality checklist

- [x] S3 integration
- [ ] Hybrid search using OpenSearch
- [x] Tests via Pytest
- [x] Swagger documentation
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
- **Local docs sub-agent** (`document_agent`) with tools to:
	- List files in the local `sagemaker-docs/` folder.
	- Transfer control back to the root agent when a request is not local-docs related.

### Example conversations

These are real-style examples based on the eval runs in `agent/evalset5d9e47.evalset.json`.

#### Example 1 — S3 bucket checks, listing, and file content

```text
User: hi!
Agent: Hello! I can help with S3 operations (bucket exists, list files, fetch content). What would you like to do?

User: Check if the bucket for sagemaker docs is created
Agent: (delegates to s3_agent) The SageMaker docs bucket exists and is accessible.

User: And what about the "opensearch-vectors" bucket? Was it created already?
Agent: (s3_agent) The "opensearch-vectors" bucket does not exist or is not accessible.

User: Ok, back to sagemaker bucket. How many texts do we have stored in it?
Agent: (s3_agent) Lists objects and summarizes the count (example output: 336 markdown files under `sagemaker-docs/`).

User: Ok, give me content for file "sagemaker-algo-common-data-formats" in this bucket then
Agent: (s3_agent) Fetches `sagemaker-docs/sagemaker-algo-common-data-formats.md` and returns its content.

User: Is there a file named "sagemaker-random-ideas" stored in this bucket?
Agent: (s3_agent) Searches by prefix and confirms no matching files.
```

Notes:

- If you omit a bucket name, the agent uses the default bucket from `S3_BUCKET_NAME`.
- For content fetches, providing the full key (e.g. `sagemaker-docs/<name>.md`) is the most reliable.

#### Example 2 — Comparing local docs vs S3 docs (agent delegation)

```text
User: hi there!
Agent: Hello! I can help with S3 bucket operations and local documentation tasks.

User: I wanna check if we have the same amount of documents in both local folder and S3 bucket for sagemaker documentation.
Agent: (root_agent) Starts with local docs.
Agent: (delegates to document_agent) Counts files in `sagemaker-docs/`.
Agent: (document_agent -> root_agent) Reports local count (example: 336).
Agent: (delegates to s3_agent) Lists objects in the configured S3 bucket.
Agent: Summarizes whether the counts match.
```

This is a good “handoff” example: the root agent coordinates the work, while sub-agents do the specialist operations and transfer control back when done.

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

## Testing

Run all tests:

```bash
pytest
```

Run just the S3 service tests:

```bash
pytest -q tests/services/test_s3_service.py
```

### Service tests (fake async client pattern)

Most services in this repo are thin wrappers around external APIs (AWS, HTTP, etc). For fast and reliable unit tests, prefer faking the dependency rather than calling the real service.

Example: `S3Service` creates its own `aioboto3` client internally. In tests we replace the private `_client()` factory with a fake async context manager that records calls and returns controlled responses.

```python
import asyncio
from app.services.config import S3Config
from app.services.s3_service import S3Service


class FakeS3Client:
	def __init__(self):
		self.calls = []
		self.get_object_response = {"Body": None}

	async def __aenter__(self):
		return self

	async def __aexit__(self, exc_type, exc, tb):
		return False

	async def get_object(self, **kwargs):
		self.calls.append(("get_object", kwargs))
		return self.get_object_response


fake = FakeS3Client()
service = S3Service(S3Config(bucket_name="unit-test-bucket"))
service._client = lambda: fake  # replace aioboto3 client factory

data = asyncio.run(service.get_file_content(key="docs/a.md"))
assert data == b""
assert fake.calls[-1][0] == "get_object"
```

Full reference tests are in `tests/services/test_s3_service.py`.
