# Plan: OpenSearch Hybrid Search Service

**TL;DR:** Build a complete OpenSearch hybrid search layer (BM25 + neural/kNN) following the project's existing architecture patterns: `Config → Models → Service → Dependencies → Routes → Tests → Shared Tools → Agent → MCP`. Documents from `sagemaker-docs/` are chunked via `DocumentService.chunk_text()` and indexed as one-document-per-chunk. The ML connector (Titan Embed v2, 1024 dims) auto-generates vectors on ingest. A dashboard setup script handles the full server-side setup (connector, model, pipelines, index creation). Since the model search returned 0 hits, the script must include connector + model registration.

## Steps

### 1. `OpenSearchConfig`

Add a frozen dataclass to `app/services/config/__init__.py` alongside `S3Config`.

- Required: `endpoint` (from `OPENSEARCH_ENDPOINT`), `index_name` (from `OPENSEARCH_INDEX_NAME`)
- Optional: `region` (falls back to `AWS_REGION`), `service_name` (default `"aoss"`), `timeout_seconds` (default `30`)
- Static `from_env()` factory, same pattern as `S3Config.from_env()`

### 2. `OpenSearchService`

New file `app/services/opensearch_service.py`.

- `_client()` factory → `OpenSearch` from `opensearch-py` with `Urllib3AWSV4SignerAuth` (SigV4, service=`aoss`). Follows the `S3Service._client()` pattern for testability.
- `document_exists(filename: str) -> bool` — Checks whether any chunks for the given filename are already indexed. Uses a `term` query on `filename` with `size=0` + `track_total_hits=true` to check count > 0 efficiently (no docs fetched, just count).
- `list_indexed_documents() -> list[str]` — Returns a deduplicated, sorted list of all `filename` values currently indexed. Uses a `terms` aggregation on `filename` (keyword sub-field).
- `index_document(filename: str, content: str, chunk_size: int, chunk_overlap: int) -> IndexDocumentResponse` — **Skips indexing if `document_exists(filename)` returns `True`** (returns response with `chunk_count=0`, `doc_ids=[]`, `skipped=True`). Otherwise calls `DocumentService.chunk_text()` on `content`, then indexes each chunk as a separate doc. Returns list of doc IDs. Each indexed document has fields: `filename`, `content` (chunk text), and `content_embedding` (auto-filled by ingest pipeline).
- `bulk_index_documents(documents: list[IndexDocumentRequest]) -> BulkIndexResponse` — Accepts a list. **For each document, checks `document_exists()` first and skips already-indexed files.** Chunks remaining documents and uses OpenSearch `_bulk` API for efficiency. This is the "parallel" indexing the user requested. Response includes which files were skipped vs indexed.
- `search(query: str, size: int = 10, search_type: Literal["hybrid", "text", "vector"] = "hybrid") -> SearchResponse` — Dispatches to the appropriate query body. **`hybrid` is the default search type.** Hybrid uses both `match` (BM25 on `content`) + `neural` (on `content_embedding`) via the search pipeline with normalization; text uses `multi_match` on `filename` + `content`; vector uses `neural` only.
- `delete_document(doc_id: str) -> bool`
- `delete_documents_by_filename(filename: str) -> int` — Deletes all chunks for a given source file (for re-indexing).
- `get_index_stats() -> IndexStatsResponse` — Document count, index health.
- Custom `OpenSearchServiceError(RuntimeError)` for consistent error handling.

### 3. Pydantic Models

New file `app/models/opensearch.py`.

- `IndexDocumentRequest(BaseModel)`: `filename: str`, `content: str`
- `IndexDocumentResponse(BaseModel)`: `filename: str`, `chunk_count: int`, `doc_ids: list[str]`, `skipped: bool = False`
- `BulkIndexRequest(BaseModel)`: `documents: list[IndexDocumentRequest]`, `chunk_size: int = 500`, `chunk_overlap: int = 50`
- `BulkIndexResponse(BaseModel)`: `total_chunks: int`, `indexed_count: int`, `skipped_count: int`, `results: list[IndexDocumentResponse]`
- `DocumentExistsResponse(BaseModel)`: `filename: str`, `exists: bool`
- `IndexedDocumentsResponse(BaseModel)`: `count: int`, `filenames: list[str]`
- `SearchRequest(BaseModel)`: `query: str`, `size: int = 10`, `search_type: Literal["hybrid", "text", "vector"] = "hybrid"`
- `SearchHit(BaseModel)`: `doc_id: str`, `score: float`, `filename: str`, `content: str`
- `SearchResponse(BaseModel)`: `query: str`, `search_type: str`, `total_hits: int`, `hits: list[SearchHit]`
- `IndexStatsResponse(BaseModel)`: `index_name: str`, `doc_count: int`, `status: str`
- `DeleteByFilenameResponse(BaseModel)`: `filename: str`, `deleted_count: int`

### 4. Dependencies

Update `app/services/dependencies.py`.

- `get_opensearch_service() -> OpenSearchService` — Constructs from `OpenSearchConfig.from_env()` + injects a `DocumentService` for chunking.
- (No setup service needed — the index/pipelines are created via the dashboard script, not at app startup. This is deliberate: serverless collections manage infra externally.)

### 5. Routes

New file `app/routes/opensearch.py`, prefix `/opensearch`, tag `opensearch`.

- `POST /opensearch/index` — Index a single document (body: `IndexDocumentRequest` + query params `chunk_size`, `chunk_overlap`). **Skips if filename already indexed** (returns `skipped=True`). Returns `IndexDocumentResponse`. Swagger examples included.
- `POST /opensearch/bulk-index` — Bulk index multiple documents (body: `BulkIndexRequest`). **Skips already-indexed filenames.** Returns `BulkIndexResponse` (includes `indexed_count` and `skipped_count`).
- `POST /opensearch/index-local-docs` — Convenience: reads ALL files from local `sagemaker-docs/` folder, bulk-indexes them. **Skips already-indexed files.** Query params for `chunk_size`, `chunk_overlap`. Good for initial data load and safe to re-run.
- `POST /opensearch/search` — Search documents (body: `SearchRequest`). **Defaults to hybrid search** when `search_type` is omitted. Returns `SearchResponse`.
- `GET /opensearch/index/stats` — Returns `IndexStatsResponse`.
- `GET /opensearch/documents` — List all indexed document filenames. Returns `IndexedDocumentsResponse`.
- `GET /opensearch/document/exists` — Check if a document filename is already indexed. Query param `filename` (required). Returns `DocumentExistsResponse`.
- `DELETE /opensearch/document/{doc_id}` — Delete single document by ID.
- `DELETE /opensearch/documents` — Delete all chunks for a given filename (query param `filename`). Returns `DeleteByFilenameResponse`.
- Wire into `main.py`: import `opensearch_router`, `app.include_router()`, add `OpenSearchServiceError` → 502 exception handler.

### 6. Tests — Service

New file `tests/services/test_opensearch_service.py`.

- `FakeOpenSearchClient` class mirroring the `FakeS3Client` pattern — tracks calls, configurable responses/exceptions.
- `StubDocumentService` for `chunk_text()` — returns predictable chunks.
- Test cases: index single doc (verify chunking + bulk call), index single doc that already exists (verify skip), bulk index multiple docs (mix of new and existing — verify correct skip/index split), document_exists returns True/False, list_indexed_documents returns deduplicated sorted filenames, search hybrid/text/vector, delete by ID, delete by filename, get stats, error handling (connection errors, auth errors, missing index), empty content edge case, empty search results, empty index (list returns []), validation errors.

### 7. Tests — Routes

New file `tests/routes/test_opensearch_routes.py`.

- `StubOpenSearchService` class following `StubS3Service` pattern.
- Update `tests/conftest.py` to include `opensearch_router` in the test app + `OpenSearchServiceError` handler.
- Test cases: successful index, index skipped (already exists), document exists (true/false), list indexed documents, successful search (all 3 types), index stats, delete, bulk index (with skips), 502 on service errors, 422 on validation errors, 404 on missing document.

### 8. Shared Tools

Update `shared/tools.py`.

- `opensearch_search(query, size, search_type="hybrid")` → calls `OpenSearchService.search()`. **Defaults to hybrid search** when `search_type` is not specified.
- `opensearch_index_document(filename, content)` → calls `OpenSearchService.index_document()` (auto-skips if already indexed)
- `opensearch_document_exists(filename)` → calls `OpenSearchService.document_exists()`
- `opensearch_list_indexed_documents()` → calls `OpenSearchService.list_indexed_documents()`
- `opensearch_get_index_stats()` → calls `OpenSearchService.get_index_stats()`
- `_get_opensearch_service()` helper following `_get_s3_service()` pattern
- `build_opensearch_tools() -> list[ToolUnion]` — packages the 3 tools + `transfer_to_root`

### 9. Agent

Update `agent/agent_factory.py`.

- New `build_opensearch_agent(settings)` function, pattern matches `build_s3_agent()`.
- Instruction: "You are the OpenSearch agent. You help the user search SageMaker documentation using hybrid search (text + vector). Use your tools to search, index documents, and check index stats."
- Add as sub-agent to `root_agent`, update root instruction to mention `opensearch_agent`.

### 10. MCP Tools

Update `mcp_server/tools.py`.

- Add `opensearch_search`, `opensearch_index_document`, `opensearch_document_exists`, `opensearch_list_indexed_documents`, `opensearch_get_index_stats` as MCP resources, wrapping `shared_tools` functions. Follow the existing `s3_*` resource pattern.

### 11. Dashboard Setup Script

New file `opensearch/setup_index.md`.

Store in a top-level `opensearch/` folder (market practice: infra scripts colocated but separate from app code).

Contents (copy-paste ready for OpenSearch Dashboard Dev Tools):

1. **Create Bedrock connector** — `POST /_plugins/_ml/connectors/_create` with `amazon.titan-embed-text-v2:0`, IAM role ARN, region.
2. **Register model** — `POST /_plugins/_ml/models/_register` using the connector ID.
3. **Deploy model** — `POST /_plugins/_ml/models/{model_id}/_deploy`
4. **Create ingest pipeline** — `PUT /_ingest/pipeline/sagemaker-docs-ingest-pipeline` with `text_embedding` processor mapping `content` → `content_embedding`.
5. **Create search pipeline** — `PUT /_search/pipeline/sagemaker-docs-search-pipeline` with `normalization-processor` (arithmetic mean of BM25 + kNN scores).
6. **Delete existing empty index** — `DELETE /sagemaker-docs`
7. **Create index with mappings** — `PUT /sagemaker-docs` with:
   - `settings.index.default_pipeline: sagemaker-docs-ingest-pipeline`
   - `settings.index.search.default_pipeline: sagemaker-docs-search-pipeline`
   - `settings.index.knn: true`
   - Mappings: `filename` (keyword + text), `content` (text), `content_embedding` (knn_vector, dim=1024, engine=faiss, space_type=l2)
8. Verification queries: `GET /sagemaker-docs/_mapping`, `GET /sagemaker-docs/_settings`

> Note: Steps 1-3 require the ConnectorID/ModelID from previous step output. The script will have placeholder comments where the user must paste the ID from the response.

### 12. README Update

Update `README.md`.

- Check `[x] Hybrid search using OpenSearch` in the checklist.
- New "OpenSearch Hybrid Search" section: explains the single-collection architecture, the ML connector, hybrid search approach.
- New API routes table for `/opensearch` endpoints.
- Update the agent section to describe `opensearch_agent`.
- Update MCP tools list.
- Add a "Why `opensearch-py`?" note explaining: `aioboto3`/`boto3` only handles control-plane (collection management), not data-plane (`_search`, `_index`); `langchain-community`'s `OpenSearchVectorSearch` doesn't support ML-connector auto-embedding and is vector-only (no native hybrid); `opensearch-py` provides direct data-plane HTTP access with SigV4 auth.
- Regarding `opensearch-py` sync vs async client: the async client has limited serverless support and is less stable with SigV4 auth. The sync client wrapped in `asyncio.to_thread()` is the most reliable approach for FastAPI routes while still maintaining responsiveness.
- Replace old references to "two OpenSearch collections (search + vector)" with the single `ragbot-v2-collection`.

### 13. Wire Up in main.py

Update `main.py`.

- Import and include `opensearch_router`.
- Add `OpenSearchServiceError` exception handler (→ 502).
- _Not_ adding OpenSearch setup to lifespan (index is managed via dashboard script).

## Verification

- `pytest` — all new + existing tests pass.
- `ruff check .` — no lint errors.
- Manual: run the dashboard script steps in OpenSearch Dev Tools, then `POST /opensearch/index-local-docs` to bulk-index all sagemaker-docs, then `POST /opensearch/search` with a query to verify hybrid results.
- Agent: `adk web --port 8001` → ask "Search for SageMaker endpoint configuration docs" → verify `opensearch_agent` is invoked and returns results.

## Key Decisions

- **One doc per chunk** (standard RAG): each chunk of a sagemaker-doc is a separate OpenSearch document, linked by `filename`.
- **No client-side embeddings for indexing**: the ML connector ingest pipeline handles embedding generation server-side. `DocumentService.embed_text()` is not used for OpenSearch indexing.
- **Dashboard script for infra, not app startup**: unlike `S3SetupService` (which auto-creates the bucket), OpenSearch index/pipeline creation is a one-time infra setup done manually via the dashboard. This avoids accidental index recreation and is standard for serverless collections.
- **`opensearch-py` sync client** (not async): `opensearch-py`'s `AsyncOpenSearch` has limited serverless support. The sync client with `Urllib3AWSV4SignerAuth` is the most stable path. Wrap calls in `asyncio.to_thread()` for non-blocking FastAPI routes.
- **Hybrid normalization**: arithmetic mean via search pipeline, combining BM25 and neural scores. This is the standard default and can be tuned later.
