# Copilot Instructions — AWS RAG Bot v2

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) backend** built with **FastAPI**, backed by **AWS S3**, **Lambda** (FAISS + BM25 hybrid search), and an **agent layer** (Google ADK + MCP). The codebase follows a strict layered architecture. All new functionality must follow the step-by-step implementation workflow described in `.github/copilot-code-instructions.md`.

## Tech Stack

- **Python 3.11** (Conda environment)
- **FastAPI** — HTTP API framework
- **Pydantic v2** — request/response models, validation
- **aioboto3 / botocore** — async AWS SDK (S3)
- **boto3** — AWS SDK (Lambda invoke, S3 operations for search)
- **faiss-cpu** — FAISS vector similarity search
- **rank-bm25** — BM25Okapi lexical search
- **LangChain** — text splitting (`RecursiveCharacterTextSplitter`), embeddings (`BedrockEmbeddings`)
- **Google ADK** — agent framework (`Agent`, `FunctionTool`, `LiteLlm`, `Runner`)
- **FastMCP** — MCP server exposing tools as resources
- **LiteLLM** — LLM routing (Bedrock Claude Sonnet 4)
- **Terraform** — infrastructure as code (Lambda, S3, IAM)
- **pytest** — testing (with `FastAPI TestClient`, fake clients, stub services)
- **ruff** — linting

## Running Tests & Linting

The Conda environment is installed locally at `.venv/` (see `prefix` in `environment.yml`). Do **not** rely on `conda activate`, `python -m ruff`, or shell-activation scripts — they are fragile and often fail on Windows.

### From a PowerShell terminal (human developer)

Invoke tools via their direct executable paths:

```powershell
# Linting
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe check shared/ mcp_server/ agent/agent_factory.py
.\.venv\Scripts\ruff.exe check . --fix

# Tests
.\.venv\Scripts\pytest.exe tests/ -v
.\.venv\Scripts\pytest.exe tests/routes/test_search_routes.py -v
```

### From Claude Code / automated tools (bash shell on Windows)

Direct `.exe` paths fail from bash with DLL errors (exit code 3228369023). Use `conda run --prefix` instead, piping output to a file:

```bash
# Tests
powershell.exe -NoProfile -Command "conda run --prefix '.venv' pytest tests/ -v 2>&1 | Out-File -FilePath 'test_results.txt' -Encoding utf8"
# Then read test_results.txt for results

# Linting
powershell.exe -NoProfile -Command "conda run --prefix '.venv' ruff check . 2>&1 | Out-File -FilePath 'lint_results.txt' -Encoding utf8"
# Then read lint_results.txt for results
```

See `docs/running-tests.md` for a full DO / DON'T reference.

## Project Structure

```
├── .github/
│   ├── copilot-instructions.md       # General Copilot instructions (this file)
│   └── copilot-code-instructions.md  # Code generation workflow & patterns
├── .githooks/
│   └── post-checkout                 # Auto-init submodules after clone/checkout
├── app/
│   ├── error_handlers.py  # Centralised exception-to-HTTP-response handlers (register_error_handlers())
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
│   └── search_tools.py        # Search tool functions + build_search_tools()
├── mcp_server/
│   ├── __init__.py            # FastMCP instance (mcp = FastMCP(...))
│   ├── main.py                # MCP server entry point (port 8002)
│   ├── s3_tools.py            # MCP resource wrappers for S3
│   ├── document_tools.py      # MCP resource wrappers for local documents
│   └── search_tools.py        # MCP resource wrappers for search
├── lambda_search/
│   ├── __init__.py
│   ├── handler.py             # Lambda function: FAISS + BM25 hybrid search
│   └── requirements.txt       # Lambda-specific dependencies
├── infra/                     # Terraform infrastructure (Lambda, S3, IAM)
├── tests/
│   ├── conftest.py        # Shared pytest fixtures (FastAPI test app, TestClient)
│   ├── services/          # Service-layer unit tests (fake client pattern)
│   └── routes/            # Route-layer tests (stub service + dependency override)
├── sagemaker-docs/        # Local markdown documentation corpus
├── experiments/           # Jupyter notebooks for search experiments
│   ├── helpers.py                    # Shared utilities for notebooks
│   ├── requirements.txt             # Experiment-only Python dependencies
│   ├── search_type_comparison.ipynb  # Experiment 1: hybrid vs text vs vector
│   └── reranking_strategies.ipynb   # Experiment 2: reranking strategies
├── plans/                 # Planning/prompt documents
├── scripts/
│   └── build_search_index.py       # Build FAISS+BM25 artifacts and upload to S3
├── archive/               # Archived scripts (e.g. former OpenSearch setup)
├── docs/
│   └── search_index_setup.md       # Search index setup guide
├── wiki/                  # GitHub Wiki (git submodule)
├── CLAUDE.md                  # Claude Code entry point (points to Copilot instruction files)
├── main.py                # FastAPI app entry point (lifespan, routers)
└── environment.yml        # Conda environment definition
```

## Coding Conventions

- **`from __future__ import annotations`** at the top of every Python file.
- **Frozen dataclasses** for configs and settings (immutability).
- **Pydantic `BaseModel`** for all request/response models.
- **Async-first** in services/routes. Wrap sync clients with `asyncio.to_thread()` when necessary.
- **Custom error classes** inheriting `RuntimeError` per service, mapped to HTTP 502 in `app/error_handlers.py`.
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
- `SEARCH_LAMBDA_FUNCTION_NAME` — Lambda function name for search
- `SEARCH_INDEX_BUCKET` — S3 bucket for FAISS/BM25 index artifacts
- `SEARCH_INDEX_PREFIX` — S3 key prefix for index artifacts
- `BM25_WEIGHT` / `VECTOR_WEIGHT` — Hybrid search fusion weights
- `SEARCH_LAMBDA_TIMEOUT_SECONDS` — Lambda invoke timeout
- `BEDROCK_EMBEDDING_MODEL_ID` — Bedrock embedding model
- `BEDROCK_EMBEDDING_DIM` — Embedding dimensions (default: 1024)
- `BEDROCK_INFERENCE_PROFILE_ID` — LLM inference profile
- `BEDROCK_MODEL_ID` — LLM model ID
- `ANTHROPIC_MODEL` — LiteLLM model string override
- `ANTHROPIC_API_KEY` — API key (if using Anthropic directly)

## Documentation & Prompt Maintenance

After **any** implementation work (new features, refactors, bug fixes, config changes, dependency updates, etc.), review and update all relevant documentation and instruction files. This applies when creating a new service layer AND when updating existing functionality — any change that affects behaviour, structure, or configuration may require documentation updates.

- **`README.md`** — Project summary, technology tables, architecture diagram, section links.
- **`wiki/`** — Detailed documentation pages (Architecture, AWS Technologies, RAG and Search, Google ADK Agent, MCP Server, API Routes, Installation, Testing).
- **`.github/copilot-instructions.md`** — Project structure, tech stack, coding conventions, env vars.
- **`.github/copilot-code-instructions.md`** — Workflow steps, pattern references, verification checklist.
- **`CLAUDE.md`** — Claude Code entry point; update if instruction file responsibilities or documentation locations change.
- **`plans/`** — Any planning/prompt documents that reference changed components.
- **`.env.example`** — If new environment variables were added or existing ones changed.

Do not leave stale references. If a file, class, route, agent, or tool was renamed, moved, or removed, update every document that mentions it.

## Infrastructure Setup

Search infrastructure (Lambda function, S3 bucket for index artifacts, IAM role) is defined as Terraform in `infra/`. Deploy with:

```bash
cd infra
terraform init
terraform apply -var="index_bucket_name=your-bucket-name"
```

Then build and upload the search index:

```bash
conda run --prefix .venv python scripts/build_search_index.py --deploy-lambda
```

The `SearchSetupService` also performs idempotent setup at app startup (checks if Lambda and artifacts exist, builds/deploys if missing, bulk-indexes local docs with dedup).
