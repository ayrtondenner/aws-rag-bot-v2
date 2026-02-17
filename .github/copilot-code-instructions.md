# Code Generation Instructions — AWS RAG Bot v2

## Step-by-Step Workflow for Implementing a New Service Layer

When adding a new capability (e.g., a new external integration, a new domain), follow these steps **in order**. Each step builds on the previous one. Do not skip steps unless explicitly noted as optional.

### Step 1 — Config

**Location:** `app/services/config/__init__.py`

- Add a new **frozen dataclass** alongside existing configs (e.g., `S3Config`).
- Load values from environment variables via a `@staticmethod from_env()` factory.
- Required fields should raise `ValueError` if the env var is missing.
- Optional fields should have sensible defaults.
- If no new configuration is required (the service reuses an existing config), skip this step.

**Pattern reference:** `S3Config` in `app/services/config/__init__.py`.

### Step 2 — Models

**Location:** `app/models/<domain>.py` (new file)

- Define all **Pydantic `BaseModel`** classes for the domain:
  - Request models (API body payloads)
  - Response models (API return types, also used by shared tools and MCP)
  - Internal data transfer objects if needed
- Use `Field(...)` with descriptions for required fields and `Field(default=...)` for optional ones.
- Include `model_validate` compatibility where dictionaries are returned from services.

**Pattern reference:** `app/models/s3.py`, `app/models/document.py`.

### Step 3 — Service

**Location:** `app/services/<domain>_service.py` (new file)

- Create the domain service class accepting a config object in `__init__`.
- Define a private `_client()` factory method for the external client (keeps construction testable — tests can replace it with a fake).
- All public methods should be `async` (use `asyncio.to_thread()` to wrap sync clients when necessary).
- Define a custom error class inheriting from `RuntimeError` (e.g., `OpenSearchServiceError(RuntimeError)`) for consistent error handling.
- Use `logging.getLogger(__name__)` and log exceptions before re-raising as service errors.
- Keep business logic in the service, not in routes or tools.

**Pattern reference:** `app/services/s3_service.py` (`S3Service`, `S3ServiceError`, `_client()` factory).

### Step 4 — Dependencies

**Location:** `app/services/dependencies.py`

- Add a new factory function: `get_<domain>_service() -> <Domain>Service`.
- Construct the service from the config's `from_env()` method.
- If the new service depends on other services (e.g., needs `DocumentService` for chunking), compose them here.
- If a one-time **setup service** is needed (auto-provisioning at startup), also add `get_<domain>_setup_service()` and a corresponding class in `app/services/setup/`.

**Pattern reference:** `get_s3_service()`, `get_s3_setup_service()` in `dependencies.py`.

### Step 5 — Routes

**Location:** `app/routes/<domain>.py` (new file)

- Create an `APIRouter` with `prefix="/<domain>"` and `tags=["<domain>"]`.
- Each endpoint should:
  - Use `Depends(get_<domain>_service)` for dependency injection.
  - Accept Pydantic request models as `Body(...)` or query params.
  - Return Pydantic response models.
  - Include `responses={}` with Swagger example payloads for the 200 case.
  - Catch `ValueError` → `HTTPException(400)` for validation errors.
- Do NOT put business logic in routes — delegate to the service.

**Wire into the app** in `main.py`:
  1. Import the router: `from app.routes.<domain> import router as <domain>_router`.
  2. Register it: `app.include_router(<domain>_router)`.
  3. Add a custom exception handler in `app/error_handlers.py` mapping `<Domain>ServiceError` → HTTP 502, and register it via `register_error_handlers()`.

**Pattern reference:** `app/routes/s3.py`, `app/routes/document.py`, and how they're wired in `main.py`.

### Step 6 — Tests

Tests follow two distinct patterns depending on the layer being tested. Write tests for both common/happy-path cases and edge cases.

#### 6a — Service Tests

**Location:** `tests/services/test_<domain>_service.py`

- Create a **fake client class** (e.g., `FakeOpenSearchClient`) that:
  - Implements the same async context manager protocol as the real client.
  - Tracks calls in a `self.calls: list[tuple[str, dict]]` list.
  - Has configurable responses and injectable exceptions.
- Create a helper `_service_with_fake_client(fake) -> <Domain>Service` that instantiates the service and monkey-patches `_client`.
- If the service depends on another service, create a **stub** for it (e.g., `StubDocumentService`) with predictable return values.
- Test cases should cover:
  - Happy-path operations (correct calls forwarded, correct responses built).
  - Error handling (service wraps exceptions into `<Domain>ServiceError`).
  - Edge cases (empty inputs, missing data, boundary values).
  - Validation errors (invalid arguments raise `ValueError`).

**Pattern reference:** `tests/services/test_s3_service.py` (`FakeS3Client`, `_service_with_fake_client`).

#### 6b — Route Tests

**Location:** `tests/routes/test_<domain>_routes.py`

- Create a **stub service class** (e.g., `StubOpenSearchService`) with:
  - Configurable return values and injectable exceptions for each method.
  - A `self.seen: dict` to capture what parameters were passed.
- Use `fastapi_app.dependency_overrides[get_<domain>_service] = lambda: stub` to inject the stub.
  - Always clean up overrides in a `finally` block or fixture teardown.
- Update `tests/conftest.py`:
  - Add the new router to the test `FastAPI` app.
  - Verify the new `<Domain>ServiceError` exception handler is registered in `app/error_handlers.py`.
- Test cases should cover:
  - Success responses (correct status code and JSON body).
  - Service errors mapped to 502.
  - Validation errors (422 from Pydantic, 400 from explicit checks).
  - Missing required parameters.

**Pattern reference:** `tests/routes/test_s3_routes.py` (`StubS3Service`, dependency override pattern), `tests/conftest.py`.

#### 6c — Other Tests

- If the domain has shared tools, add tests in `tests/shared/`.
- If the domain has complex model logic, add model-level tests in `tests/models/`.

### Step 7 — Shared Tools (ADK)

**Location:** `shared/<domain>_tools.py` (new file per domain, e.g., `shared/s3_tools.py`)

- Cross-cutting helpers (`transfer_to_root`, `DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME`) live in `shared/__init__.py`.
- Add async tool functions that wrap the service methods (e.g., `opensearch_query(...)`, `opensearch_index_document(...)`).
- Add a private helper `_get_<domain>_service()` following the `_get_s3_service()` pattern.
- Add a `build_<domain>_tools() -> list[ToolUnion]` function that returns:
  - `FunctionTool(...)` for each tool function.
  - `FunctionTool(transfer_to_root)` so the sub-agent can hand back control.
- Tool functions should have clear docstrings (ADK uses these for the LLM's tool descriptions).
- Return the same Pydantic response models used by routes for consistency.

**Pattern reference:** `shared/s3_tools.py` (`build_s3_tools()`), `shared/document_tools.py` (`build_document_tools()`).

### Step 8 — Agent (ADK)

**Location:** `agent/agent_factory.py`

- Add a new `build_<domain>_agent(settings: Settings) -> Agent` function following the same pattern as `build_s3_agent` and `build_document_agent`.
- The agent should have:
  - A clear `description` (used by the root agent for delegation).
  - An `instruction` (system prompt telling the LLM its role and when to transfer back).
  - `tools=build_<domain>_tools()` from `shared/<domain>_tools.py`.
- Register the new agent as a sub-agent of `root_agent`:
  - Add it to the `sub_agents=[...]` list.
  - Update the root agent's `instruction` string to mention the new sub-agent and when to delegate to it.

**Pattern reference:** `build_s3_agent()`, `build_document_agent()` in `agent/agent_factory.py`.

### Step 9 — MCP Tools

**Location:** `mcp_server/<domain>_tools.py` (new file per domain, e.g., `mcp_server/s3_tools.py`)

The `FastMCP` instance lives in `mcp_server/__init__.py`. Each domain tool module imports it and registers resources with `@mcp.resource(...)` decorators. `mcp_server/main.py` performs side-effect imports of all tool modules to ensure resources are registered before the server starts.

- Add MCP resource (or tool) wrappers that delegate to the corresponding `shared.<domain>_tools` functions.
- Use `@mcp.resource(...)` with:
  - `name` — snake_case identifier.
  - `description` — matches the shared tool's docstring.
  - `uri` — follow existing URI scheme patterns (e.g., `s3://...`, `local://...`, `opensearch://...`).
- Use `Annotated[<type>, Field(...)]` for parameter metadata.
- Return the same Pydantic response models as the shared tools.
- After creating a new tool module, add a side-effect import in `mcp_server/main.py` (e.g., `import mcp_server.<domain>_tools  # noqa: F401`).

**Pattern reference:** `mcp_server/s3_tools.py`, `mcp_server/document_tools.py`, `mcp_server/opensearch_tools.py`.

---

## Verification Checklist

After implementing all steps, confirm:

1. **`pytest`** — all new and existing tests pass.
2. **`..\.venv\Scripts\ruff.exe check .`** — no lint errors (always invoke ruff via its direct executable path; do not use `conda activate`, `python -m ruff`, or bare `ruff` commands).
3. **Swagger UI** (`/docs`) — new endpoints appear with correct examples.
4. **Manual test** — hit the new routes (e.g., via `curl` or Swagger) for smoke testing.
5. **Agent test** — run `adk web --port 8001` and verify the root agent delegates to the new sub-agent correctly.
6. **MCP test** — verify MCP resources are listed and callable.
7. **Documentation** — update `README.md`, `.github/copilot-instructions.md`, `.github/copilot-code-instructions.md`, `prompts/`, and `.env.example` to reflect all changes (new routes, agents, tools, env vars, structure changes, etc.). See the "Documentation & Prompt Maintenance" section in `.github/copilot-instructions.md`.
