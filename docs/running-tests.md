# Running Python Tests & Linting

This project uses a **Conda prefix environment** at `.venv/`. The correct way to run tools depends on your shell context.

## DO

### PowerShell terminal (human developer)

```powershell
# Tests
.\.venv\Scripts\pytest.exe tests/ -v
.\.venv\Scripts\pytest.exe tests/routes/test_opensearch_routes.py -v

# Linting
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe check . --fix
```

### Claude Code / automated tools (bash shell on Windows)

Use `conda run --prefix` and redirect output to a file:

```bash
# Tests
powershell.exe -NoProfile -Command "conda run --prefix '.venv' pytest tests/ -v 2>&1 | Out-File -FilePath 'test_results.txt' -Encoding utf8"
# Then read test_results.txt

# Specific test file
powershell.exe -NoProfile -Command "conda run --prefix '.venv' pytest tests/routes/test_opensearch_routes.py -v 2>&1 | Out-File -FilePath 'test_results.txt' -Encoding utf8"

# Linting
powershell.exe -NoProfile -Command "conda run --prefix '.venv' ruff check . 2>&1 | Out-File -FilePath 'lint_results.txt' -Encoding utf8"
# Then read lint_results.txt
```

## DON'T

| Command | Why it fails |
|---------|-------------|
| `.venv/Scripts/pytest.exe` from bash | DLL error (exit code 3228369023 / 0xC06D007F) — conda DLLs not on PATH |
| `.venv/python.exe -m pytest` from bash | Same DLL error — the Python executable itself needs conda's DLL paths |
| `conda activate .venv` | Fragile on Windows, requires shell integration that isn't available in all contexts |
| `conda run -n aws-rag-bot` | This is a prefix install, not a named environment — `-n` won't find it |
| `python -m pytest` (system Python) | Uses system Python (3.14+), which doesn't have project dependencies installed |

## Why `conda run --prefix` works

`conda run --prefix .venv` sets up the full conda environment (including DLL search paths like `Library/bin`) before executing the command. This is the only reliable way to run conda-installed tools from a non-conda shell.
