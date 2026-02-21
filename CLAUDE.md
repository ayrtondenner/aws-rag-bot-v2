# Claude Code Instructions — AWS RAG Bot v2

This project uses **GitHub Copilot instruction files** as the single source of truth for
coding conventions, architectural patterns, and the code-generation workflow. Claude Code
should treat those files as authoritative and apply their rules to all tasks.

## Instruction Files

| File | Responsibility |
|------|----------------|
| `.github/copilot-instructions.md` | Always-on workspace instructions: project overview, tech stack, coding conventions, naming conventions, environment variables, error-handling patterns, documentation standards, and the project structure tree. |
| `.github/copilot-code-instructions.md` | Code-generation workflow: step-by-step implementation patterns for adding new service layers (Config → Models → Service → Dependencies → Routes → Tests → Shared Tools → Agent → MCP Tools), plus the verification checklist. |

**Read both files before starting any implementation task.**

## Documentation Locations

Project documentation is divided across three locations. Before starting a task that
involves documentation — or whenever a task description mentions documentation — check
all three:

| Location | Contents |
|----------|----------|
| `README.md` | Project summary, architecture diagram, technology tables, section links to the wiki. |
| `wiki/` | Nine detailed wiki pages (Architecture, AWS Technologies, RAG and OpenSearch, Google ADK Agent, MCP Server, API Routes, Installation, Testing, Home). This folder is a git submodule that mirrors the GitHub Wiki. |
| `.github/copilot-instructions.md` and `.github/copilot-code-instructions.md` | Coding conventions, patterns, and the workflow checklist. |

After any implementation work, review all three locations and update whichever pages are
affected. See the "Documentation & Prompt Maintenance" section in
`.github/copilot-instructions.md` for the complete checklist.

## Running Tests & Linting

The Conda environment is installed at `.venv/` as a **local prefix** (not a named env).
Running `.exe` files directly from a non-conda shell (e.g., bash inside VSCode) fails with
DLL errors. Always use `conda run --prefix`:

```powershell
# Run all tests
conda run --prefix .venv pytest tests/ -v

# Run a specific test file
conda run --prefix .venv pytest tests/routes/test_opensearch_routes.py -v

# Lint
conda run --prefix .venv ruff check .

# Lint with auto-fix
conda run --prefix .venv ruff check . --fix
```

When running from **Claude Code's bash shell**, pipe through `powershell.exe` and redirect
output to a file (stdout doesn't pipe cleanly through bash → powershell):

```bash
powershell.exe -NoProfile -Command "conda run --prefix '.venv' pytest tests/ -v 2>&1 | Out-File -FilePath 'test_results.txt' -Encoding utf8"
# Then read test_results.txt
```

See `docs/running-tests.md` for a full DO / DON'T reference.
