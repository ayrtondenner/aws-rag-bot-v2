# Phase 1: Stabilization & Architecture Cleanup — Epics

> Source: GitHub Issues #10, #17, #18, #19
> Sprint: Phase 1 (post-FAISS migration)

---

## Epic 1: MCP Server Rework

Rework MCP server to correctly classify tools vs resources and expose missing delete operations.

### Story 1.1: Reclassify search_index_document as MCP Tool

Change `search_index_document` from `@mcp.resource` to `@mcp.tool` since it performs a write operation (indexes documents into FAISS+BM25). Remove the `uri` parameter.

**GitHub Issue:** #10

### Story 1.2: Add Delete Operations to MCP and Shared Tools

Add `search_delete_document` and `search_delete_by_filename` as `@mcp.tool` in the MCP server, with corresponding wrappers in `shared/search_tools.py` and ADK tool registration.

**GitHub Issue:** #10

### Story 1.3: Add MCP Tool/Resource Classification Tests

Create `tests/mcp_server/test_search_tools.py` to verify that write operations are registered as tools and read-only operations remain resources.

**GitHub Issue:** #10

---

## Epic 2: Lambda Build Index and FastAPI Delegation

Create a new Lambda `build_index` action and refactor `SearchSetupService` to delegate artifact building to Lambda instead of doing it locally.

### Story 2.1: Implement Lambda build_index Action

Add `_handle_build_index()` to `lambda_search/handler.py` that accepts raw documents, chunks them, embeds via Bedrock, builds FAISS+BM25 indexes, and saves to S3. Support `skip_existing` dedup.

**GitHub Issues:** #17, #18

### Story 2.2: Add SearchService.build_index Method

Add `build_index()` async method to `SearchService` that sends documents to Lambda in batches and aggregates results.

**GitHub Issues:** #17, #18

### Story 2.3: Refactor SearchSetupService to Delegate to Lambda

Replace `_build_and_upload_artifacts()` in `SearchSetupService` with an async method that reads local docs and delegates to `SearchService.build_index()`. Remove local FAISS/BM25 building code.

**GitHub Issues:** #17, #18

### Story 2.4: Update Lambda Timeout and Infrastructure

Change `lambda_timeout` in `infra/variables.tf` from 30s to 300s to accommodate bulk index building.

**GitHub Issues:** #17, #18

### Story 2.5: Add Tests for Lambda build_index and SearchService.build_index

Add tests for the new Lambda action and SearchService method. Update existing setup service tests.

**GitHub Issues:** #17, #18

---

## Epic 3: Move Chunking and Dedup to Lambda

Extend Lambda `index_documents` to handle raw content and dedup, simplify `SearchService` by removing `DocumentService` dependency.

### Story 3.1: Extend Lambda index_documents with Content and Skip-Existing

Extend `_handle_index_documents()` in Lambda to accept `content` (raw text) alongside `chunks` (backward compat). Add `skip_existing` flag for dedup.

**GitHub Issue:** #19

### Story 3.2: Simplify SearchService.index_document

Remove `document_exists` round-trip and local `chunk_text` call. Send raw content to Lambda with `skip_existing=True`. Remove `DocumentService` dependency from `SearchService.__init__`.

**GitHub Issue:** #19

### Story 3.3: Update Dependencies and Route Handlers

Update `get_search_service()` in `dependencies.py` to stop passing `document_service`. Simplify `_index_local_docs()` flow.

**GitHub Issue:** #19

### Story 3.4: Rework SearchService Tests

Major rework of `test_search_service.py`: remove `StubDocumentService`, update index tests to expect single Lambda calls with content field, add backward-compat tests for Lambda handler.

**GitHub Issue:** #19
