# Copilot Instructions — AWS RAG Bot v2

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) backend** built with **FastAPI**, backed by **AWS S3**, **Amazon OpenSearch** (hybrid search), and an **agent layer** (Google ADK + MCP). The codebase follows a strict layered architecture. All new functionality must follow the step-by-step implementation workflow described in `.github/copilot-code-instructions.md`.

## Tech Stack

- **Python 3.11** (Conda environment)
- **FastAPI** — HTTP API framework
- **Pydantic v2** — request/response models, validation
- **aioboto3 / botocore** — async AWS SDK (S3)
- **opensearch-py** — OpenSearch data-plane client (sync, wrapped with `asyncio.to_thread()`)
- **LangChain** — text splitting (`RecursiveCharacterTextSplitter`), embeddings (`BedrockEmbeddings`)
- **Google ADK** — agent framework (`Agent`, `FunctionTool`, `LiteLlm`, `Runner`)
- **FastMCP** — MCP server exposing tools as resources
- **LiteLLM** — LLM routing (Bedrock Claude Sonnet 4)
- **pytest** — testing (with `FastAPI TestClient`, fake clients, stub services)
- **ruff** — linting

## Running Ruff

The Conda environment is installed locally at `.venv/` (see `prefix` in `environment.yml`). Do **not** rely on `conda activate`, `python -m ruff`, or shell-activation scripts — they are fragile and often fail on Windows.

**Always invoke ruff via its direct executable path:**

```powershell
.\.\.venv\Scripts\ruff.exe check .
```

For targeted checks on specific directories/files:

```powershell
.\.\.venv\Scripts\ruff.exe check shared/ mcp_server/ agent/agent_factory.py
```

To auto-fix fixable issues:

```powershell
.\.\.venv\Scripts\ruff.exe check . --fix
```

## Project Structure

```
├── .github/
│   ├── copilot-instructions.md       # General Copilot instructions (this file)
│   └── copilot-code-instructions.md  # Code generation workflow & patterns
├── app/
│   ├── models/            # Pydantic request/response models
│   ├── routes/            # FastAPI route modules (one per domain)
│   ├── services/
│   │   ├── config/        # Frozen dataclass configs loaded from env vars
│   │   ├── setup/         # One-time infrastructure provisioning services
│   │   ├── dependencies.py  # FastAPI dependency providers (factory functions)
│   │   └── *_service.py   # Domain service classes
├── agent/
│   ├── agent_factory.py   # Agent builder functions (root + sub-agents)
│   ├── agent.py           # ADK entry point (root_agent symbol)
│   ├── runtime.py         # Session/runner initialization
│   └── settings.py        # Agent settings (frozen dataclass)
├── shared/
│   ├── __init__.py            # Cross-cutting helpers (transfer_to_root, DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME)
│   ├── s3_tools.py            # S3 tool functions + build_s3_tools()
│   ├── document_tools.py      # Local-document tool functions + build_document_tools()
│   └── opensearch_tools.py    # OpenSearch tool functions + build_opensearch_tools()
├── mcp_server/
│   ├── __init__.py            # FastMCP instance (mcp = FastMCP(...))
│   ├── main.py                # MCP server entry point (port 8002)
│   ├── s3_tools.py            # MCP resource wrappers for S3
│   ├── document_tools.py      # MCP resource wrappers for local documents
│   └── opensearch_tools.py    # MCP resource wrappers for OpenSearch
├── tests/
│   ├── conftest.py        # Shared pytest fixtures (FastAPI test app, TestClient)
│   ├── services/          # Service-layer unit tests (fake client pattern)
│   ├── routes/            # Route-layer tests (stub service + dependency override)
│   └── shared/            # Tests for shared tool functions
├── sagemaker-docs/        # Local markdown documentation corpus
├── prompts/               # Planning/prompt documents
├── scripts/               # Utility/smoke-test scripts
├── docs/
│   └── opensearch_index_setup.md     # Dashboard Dev Tools commands for one-time OpenSearch setup
├── main.py                # FastAPI app entry point (lifespan, routers, exception handlers)
└── environment.yml        # Conda environment definition
```

## Coding Conventions

- **`from __future__ import annotations`** at the top of every Python file.
- **Frozen dataclasses** for configs and settings (immutability).
- **Pydantic `BaseModel`** for all request/response models.
- **Async-first** in services/routes. Wrap sync clients with `asyncio.to_thread()` when necessary.
- **Custom error classes** inheriting `RuntimeError` per service, mapped to HTTP 502 in `main.py`.
- **Dependency injection** via FastAPI `Depends()` — never instantiate services directly in routes.
- **Private `_client()` factory** in services for testability (fake/stub replacement in tests).
- **`logging.getLogger(__name__)`** for all log statements.
- **Type hints everywhere** — all function signatures (parameters and return types), class attributes, and variables must be fully typed. Use `Optional[X]` for nullable parameters (preferred over `X | None`), `list[X]`, `dict[K, V]` for collections, and `Annotated[X, Field(...)]` for Pydantic/MCP parameter metadata. Import from `typing` (`Optional`, `Any`, `Literal`, etc.) and `typing_extensions` when needed. Never leave a function without a return type annotation (use `-> None` for void functions).
- **Snake_case** for files, functions, variables. **PascalCase** for classes.
- **Docstrings** on all public functions (Google style with Args/Returns sections).
- Tests use `asyncio.run()` for async service tests and `TestClient` for route tests.
- Clean up `dependency_overrides` in test teardown (use `try/finally`).

## Environment Variables

New services should load configuration from environment variables. Document new env vars in `.env.example` and in the README. Current key variables:

- `S3_BUCKET_NAME` — Default S3 bucket
- `AWS_REGION` / `AWS_DEFAULT_REGION` — AWS region
- `OPENSEARCH_ENDPOINT` — OpenSearch collection endpoint
- `OPENSEARCH_INDEX_NAME` — OpenSearch index name
- `BEDROCK_EMBEDDING_MODEL_ID` — Bedrock embedding model
- `BEDROCK_EMBEDDING_DIM` — Embedding dimensions (default: 1024)
- `BEDROCK_INFERENCE_PROFILE_ID` — LLM inference profile
- `BEDROCK_MODEL_ID` — LLM model ID
- `ANTHROPIC_MODEL` — LiteLLM model string override
- `ANTHROPIC_API_KEY` — API key (if using Anthropic directly)

## Documentation & Prompt Maintenance

After **any** implementation work (new features, refactors, bug fixes, config changes, dependency updates, etc.), review and update all relevant documentation and instruction files:

- **`README.md`** — API routes tables, agent capabilities, MCP tools list, environment variables, examples.
- **`.github/copilot-instructions.md`** — Project structure, tech stack, coding conventions, env vars.
- **`.github/copilot-code-instructions.md`** — Workflow steps, pattern references, verification checklist.
- **`prompts/`** — Any planning/prompt documents that reference changed components.
- **`.env.example`** — If new environment variables were added or existing ones changed.

Do not leave stale references. If a file, class, route, agent, or tool was renamed, moved, or removed, update every document that mentions it.

## Infrastructure Setup

Some services require one-time infrastructure setup (e.g., OpenSearch pipelines, index creation). These should be documented as runnable scripts or dashboard instructions in a dedicated folder (e.g., `docs/opensearch_index_setup.md`), **not** embedded in application startup code — unless the provisioning is idempotent and lightweight (like `S3SetupService` creating a bucket if absent).
