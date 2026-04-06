# AWS RAG Bot v2 — Roadmap & Skills Guide

> Created: 2026-04-05 | Branch at time of writing: `feat/migrate-opensearch-to-faiss-lambda`

## Overview

This document maps every open GitHub issue to a phased execution sequence and recommends
which BMAD skills, Superpowers skills, and plugins to invoke at each step. Designed for
solo-dev workflow.

---

## Phase 0: Merge the FAISS Migration Branch

**Goal:** Validate the migration, fix any test failures, and merge to main.

| Step | Issue | Skill(s) | What to do |
|------|-------|----------|------------|
| 0.1 | **#15** Fix failing tests | `superpowers:systematic-debugging` | Run tests, diagnose failures with structured debugging cycle. Then `simplify` on fixes. |
| 0.2 | **#6** Validate FAISS+BM25 config in AWS | `superpowers:verification-before-completion` | `terraform plan` for drift, invoke Lambda with test payload, verify S3 artifacts + embedding dims. |
| 0.3 | **#7** Verify startup indexing in AWS | `superpowers:verification-before-completion` | Start API against real AWS, check logs for "Search setup complete", hit `GET /search/index/stats`. |
| 0.4 | **#9** Confirm ADK+MCP hybrid search | `superpowers:verification-before-completion` | Already done in migration. Verify via `adk web` + MCP smoke test, then close issue. |
| 0.5 | — Merge to main | `superpowers:finishing-a-development-branch` then `code-review:code-review` | PR review + structured merge flow. Close issues #6, #7, #9, #15. |

**Key files:**
- `tests/` — all test files
- `app/services/setup/search_setup_service.py` — startup indexing
- `lambda_search/handler.py` — Lambda function
- `infra/` — Terraform definitions

---

## Phase 1: Stabilization & Architecture Cleanup

**Goal:** Clean up tech debt from the migration and improve architecture.

Start with: `bmad-sprint-planning` (Bob) to sequence into a formal sprint plan.

| Step | Issue | Skill(s) | What to do |
|------|-------|----------|------------|
| 1.1 | **#17** Improve artifact idempotency | `superpowers:brainstorming` → `superpowers:test-driven-development` | Explore strategies (checksumming, incremental updates), write failing tests first, then implement. |
| 1.2 | **#18** Remove local artifact building | `feature-dev:feature-dev` | Guided refactor: `SearchSetupService` delegates all indexing to Lambda instead of building locally. |
| 1.3 | **#19** Audit FastAPI work for Lambda | `bmad-agent-architect` (Winston) → `bmad-review-adversarial-general` | Architectural audit of service boundaries. Produce ADR on what moves to Lambda (chunking? dedup?). |
| 1.4 | **#10** Rework MCP tools vs resources | `bmad-technical-research` → `superpowers:brainstorming` → `superpowers:executing-plans` | Research MCP spec. Write ops → `@mcp.tool`. Read-only ops → `@mcp.resource`. |

**Key files:**
- `app/services/setup/search_setup_service.py` — #17, #18
- `lambda_search/handler.py` — #18, #19
- `app/services/search_service.py` — #19 (chunking+dedup before Lambda invoke)
- `mcp_server/search_tools.py`, `mcp_server/s3_tools.py`, `mcp_server/document_tools.py` — #10

---

## Phase 2: Infrastructure & DevOps

**Goal:** Improve developer experience and deployment story.

| Step | Issue | Skill(s) | What to do |
|------|-------|----------|------------|
| 2.1 | **#14** Containerize with Docker | `bmad-agent-architect` (Winston) → `bmad-quick-dev` (Barry) | Decide: single vs multi-stage, compose strategy for API/Agent/MCP. Then rapid implementation. |
| 2.2 | **#13** S3 file watcher | `superpowers:brainstorming` → `superpowers:test-driven-development` → `bmad-dev-story` | Explore approaches (watchdog, polling, S3 events). TDD for edge cases. |

---

## Phase 3: Experiments & Research

**Goal:** Validate design decisions and explore future directions.

| Step | Issue | Skill(s) | What to do |
|------|-------|----------|------------|
| 3.1 | **#8** Run Jupyter experiments | `superpowers:verification-before-completion` → `bmad-agent-tech-writer` (Paige) | Execute notebooks against live Lambda, fill TODO observations, polish into wiki. |
| 3.2 | **#21** Compare with OpenRag | `bmad-technical-research` → `bmad-agent-analyst` (Mary) | Structured tech comparison report with go/no-go recommendation. |
| 3.3 | **#20** Agentic RAG evaluation | `bmad-agent-pm` (John) → `bmad-check-implementation-readiness` | **Possibly already done.** Current root + 3 sub-agents IS agentic RAG. Validate or identify gaps. |

---

## Issue Priority Map

| Priority | Issues | Phase |
|----------|--------|-------|
| **Critical** | #15, #6, #7, #9 + merge | Phase 0 |
| **High** | #17, #18, #19, #10 | Phase 1 |
| **Medium** | #14, #13 | Phase 2 |
| **Medium** | #8 | Phase 3 |
| **Low** | #21, #20 (possibly done) | Phase 3 |

---

## Quick Skill Decision Tree

Use this when starting any task to pick the right workflow:

| Situation | Skill(s) |
|-----------|----------|
| Fix a bug | `superpowers:systematic-debugging` → `superpowers:test-driven-development` |
| Build a feature | `superpowers:brainstorming` → `superpowers:writing-plans` → `superpowers:test-driven-development` → `superpowers:executing-plans` |
| Architectural decision | `bmad-agent-architect` (Winston) → `bmad-review-adversarial-general` |
| Refactor code | `superpowers:writing-plans` → `simplify` → `superpowers:verification-before-completion` |
| Update docs | `bmad-agent-tech-writer` (Paige) → `bmad-editorial-review-prose` |
| Plan a sprint | `bmad-sprint-planning` (Bob) → `bmad-create-story` per item |
| Research a technology | `bmad-technical-research` → `bmad-agent-analyst` (Mary) |
| Finish a branch | `superpowers:verification-before-completion` → `superpowers:requesting-code-review` → `superpowers:finishing-a-development-branch` |
| Review test quality | `bmad-tea` (Murat) → `bmad-testarch-test-review` → `bmad-testarch-trace` |

---

## Documentation Maintenance Cadence

### After every PR merge

| Action | Skill |
|--------|-------|
| Update affected docs (README, wiki, .github/) | `bmad-agent-tech-writer` (Paige) |
| Structural review of changed docs | `bmad-editorial-review-structure` |
| Regenerate project context for AI assistants | `bmad-generate-project-context` |

### Monthly health check

| Action | Skill |
|--------|-------|
| Full doc audit for staleness | `bmad-document-project` |
| Regenerate doc indexes | `bmad-index-docs` |
| Review copilot instructions vs reality | `bmad-review-adversarial-general` |
| Sprint retrospective | `bmad-retrospective` |

---

## Verification Checklist (run after each phase)

1. `conda run --prefix .venv pytest tests/ -v`
2. `conda run --prefix .venv ruff check .`
3. Verify Swagger UI at `/docs`
4. For AWS changes: `terraform -chdir=infra plan`
5. For MCP changes: `scripts/mcp_smoke_test.py`
6. For agent changes: `adk web --port 8001`
7. Update documentation per cadence above

---

## Installed Plugins & Skills Reference

**Plugins** (in `.claude/settings.json`):
- `superpowers@claude-plugins-official` v5.0.7 — workflow skills (debugging, TDD, plans, branching)
- `code-review@claude-plugins-official` — PR code review
- `code-simplifier@claude-plugins-official` v1.0.0 — code quality review
- `feature-dev@claude-plugins-official` — guided feature development

**BMAD** (53 skills in `.claude/skills/`):
- Agents: analyst (Mary), architect (Winston), dev (Amelia), PM (John), QA (Quinn), solo-dev (Barry), scrum master (Bob), tech writer (Paige), UX (Sally), TEA (Murat)
- Planning: create-prd, create-architecture, create-epics-and-stories, create-story, sprint-planning, sprint-status
- Implementation: dev-story, quick-dev, code-review
- Documentation: document-project, generate-project-context, editorial-review-prose, editorial-review-structure
- Testing: testarch-test-design, testarch-automate, testarch-test-review, testarch-trace, testarch-framework, testarch-ci, testarch-nfr, testarch-atdd, qa-generate-e2e-tests
- Research: domain-research, technical-research, market-research
- Quality: check-implementation-readiness, validate-prd, review-adversarial-general, review-edge-case-hunter
