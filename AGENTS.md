# Repository Guidelines

## Project Structure & Module Organization
`app/` contains the FastAPI API: `routes/` for HTTP endpoints, `models/` for Pydantic schemas, and `services/` for business logic plus startup/setup helpers. `agent/` holds the Google ADK agent entry points, `shared/` contains reusable tool wrappers, and `mcp_server/` exposes the same capabilities through FastMCP. `lambda_search/` packages the FAISS + BM25 search Lambda, `infra/` stores Terraform for S3/IAM/Lambda, and `scripts/` includes operational helpers such as `build_search_index.py`. Tests live under `tests/routes/` and `tests/services/`. The local document corpus is `sagemaker-docs/`; longer references live in `docs/` and `wiki/`.

## Build, Test, and Development Commands
Create the environment with `conda env create -f environment.yml`. Run the API with `conda run --prefix .venv uvicorn main:app --reload` and open `/docs` for Swagger. Start the agent with `adk web --port 8001` and the MCP server with `python -m mcp_server.main`. Run all tests with `conda run --prefix .venv pytest tests/ -v`. Lint with `conda run --prefix .venv ruff check .`. For infrastructure changes, use `terraform -chdir=infra init` and `terraform -chdir=infra apply -var="index_bucket_name=<bucket>"`.

## Coding Style & Naming Conventions
Target Python 3.11 and use 4-space indentation. New Python modules should start with `from __future__ import annotations`. Use `snake_case` for files, functions, and variables; `PascalCase` for classes. Keep routes thin, place business logic in services, and inject dependencies through `app/services/dependencies.py` instead of constructing services inside route handlers. Prefer frozen dataclasses for config, Pydantic models for request/response payloads, explicit type hints on all signatures, and `logging.getLogger(__name__)` for logging. Ruff is the formatting/lint gate.

## Testing Guidelines
Pytest is configured through `pytest.ini` with `tests/` as the test root and `test_*.py` naming. Service tests use fake clients and async execution; route tests use `FastAPI` `dependency_overrides` with stub services and must clean overrides in teardown. Keep new tests next to the layer they verify, and cover happy paths plus validation and service-error cases. Use markers such as `@pytest.mark.smoke` or `@pytest.mark.regression` when appropriate.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `Migrate from OpenSearch Serverless to Lambda + FAISS + BM25 + S3`. Keep commits focused and descriptive. PRs should summarize behavior changes, note AWS or environment-variable impacts, link the relevant issue or plan, and list the verification performed (`pytest`, `ruff`, manual API/agent/MCP checks as applicable). Update `README.md`, `docs/`, `wiki/`, and `.env.example` whenever behavior or configuration changes.
