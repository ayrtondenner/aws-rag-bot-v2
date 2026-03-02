# OpenSearch Serverless Alternatives — Comparison Report

> **Date**: 2026-03-02
> **Context**: [Issue #N — Evaluate alternatives to replace OpenSearch Serverless for RAG hybrid search](https://github.com/ayrtondenner/aws-rag-bot-v2/issues)

---

## 1. Problem Summary

The project currently uses **AWS OpenSearch Serverless** for RAG hybrid search
(BM25 + neural vector). A cost analysis shows the service charges **~US$536/month**
in idle OCU-hours — even with near-zero traffic.

**Current usage profile:**

| Metric | Value |
|--------|-------|
| Source documents | 375 Markdown files (SageMaker CloudFormation docs) |
| Chunks after splitting | ~1k–2k (500-char chunks, 50-char overlap) |
| Embedding model | Amazon Titan Embed Text v2 (1024 dims) |
| Vector engine | FAISS HNSW (L2 distance) |
| Search weights | 0.3 BM25 + 0.7 neural (arithmetic mean normalization) |
| Weekly query volume | Dozens (testing only) |
| Region | us-east-1 |

At this scale OpenSearch Serverless is vastly over-provisioned.

---

## 2. Current Architecture Touchpoints

Before evaluating alternatives, it helps to understand exactly what needs to
change. The OpenSearch integration spans these layers:

| Layer | File(s) | Responsibility |
|-------|---------|----------------|
| **Config** | `app/services/config/__init__.py` | `OpenSearchConfig` dataclass (endpoint, index, region, service name) |
| **Service** | `app/services/opensearch_service.py` | `OpenSearchService` — index, search, delete, stats; SigV4 auth; sync `opensearch-py` client wrapped with `asyncio.to_thread()` |
| **Dependency** | `app/services/dependencies.py` | `get_opensearch_service()` factory |
| **Routes** | `app/routes/opensearch.py` | REST endpoints (`POST /search`, `POST /index`, `DELETE`, `GET /stats`) |
| **Models** | `app/models/` | Pydantic request/response models (`SearchRequest`, `SearchResponse`, `SearchHit`, etc.) |
| **Shared tools** | `shared/opensearch_tools.py` | `opensearch_query()` function used by ADK agent |
| **Agent** | `agent/agent_factory.py` | `build_opensearch_agent()` — sub-agent wrapping the shared tools |
| **MCP** | `mcp_server/opensearch_tools.py` | FastMCP wrappers for OpenSearch operations |
| **Setup** | `app/services/setup/` | Bulk-index local docs on startup |
| **Tests** | `tests/services/test_opensearch_service.py`, `tests/routes/test_opensearch_routes.py` | Unit tests with fake client / stub service |
| **Infra docs** | `docs/opensearch_index_setup.md` | Dashboard Dev Tools commands for index/pipeline creation |
| **Dependencies** | `environment.yml` | `opensearch-py`, `requests-aws4auth` |

A replacement must satisfy the same service interface (`search`, `index_document`,
`bulk_index`, `delete_document`, `get_index_stats`) and the hybrid search contract
(BM25 + vector fusion with configurable weights).

---

## 3. Option A — Lambda/Fargate + FAISS + BM25 + S3

### How It Works

1. **Offline pipeline**: chunk documents → generate embeddings (Bedrock Titan v2)
   → build a FAISS index + a BM25 index (e.g., `rank_bm25` library) → serialize
   both artifacts to **S3**.
2. **Query time**: a **Lambda function** (or **Fargate on-demand task**) downloads
   the artifacts from S3 into memory, runs hybrid search (FAISS cosine/L2 +
   BM25 scoring), applies weight fusion (0.3/0.7), returns top-K.

**Fargate variant**: keep the current FastAPI app running on Fargate Spot with
scale-to-zero (ECS + Application Auto Scaling), so no Lambda adapter is needed.

### Cost Estimate

| Component | Monthly cost |
|-----------|-------------|
| S3 storage (~10 MB artifacts) | < US$0.01 |
| S3 GET requests (~200/month) | < US$0.01 |
| Lambda invocations (200 × 512 MB × 2 s) | ~US$0.01 |
| Bedrock embedding (index-time only, ~2k calls once) | ~US$0.02 (one-time) |
| **Total** | **~US$0.05/month** |

With Fargate Spot (scale-to-zero):

| Component | Monthly cost |
|-----------|-------------|
| Fargate Spot 0.25 vCPU / 0.5 GB × ~2 hrs/month | ~US$0.02 |
| S3 storage + requests | < US$0.01 |
| **Total** | **~US$0.05/month** |

### Hybrid Search Quality

| Aspect | Assessment |
|--------|-----------|
| BM25 scoring | `rank_bm25` library provides Okapi BM25 — same algorithm as OpenSearch |
| Vector similarity | FAISS supports L2, inner product, cosine — same space types as current index |
| Fusion | Implement Reciprocal Rank Fusion (RRF) or weighted arithmetic mean manually; full control over weights |
| Reranking | Can add cross-encoder reranking on top (same as current experiments) |
| **Relevance parity** | ✅ Achievable — same embeddings, same BM25, same fusion; may even improve with RRF |

### Engineering Effort

| Task | Estimate |
|------|----------|
| Build offline index pipeline (FAISS + BM25 serialization to S3) | 2–3 days |
| Implement `HybridSearchService` (load artifacts, score, fuse, return) | 2–3 days |
| Wire into FastAPI dependencies, routes, agent, MCP | 1 day |
| Lambda packaging / Fargate task definition | 1–2 days |
| Tests (unit + integration) | 1–2 days |
| **Total** | **7–11 days** |

### Pros / Cons

| Pros | Cons |
|------|------|
| True pay-per-use; near-zero cost at current scale | More custom code to build and maintain |
| Full control over scoring, fusion, reranking | Cold start latency (Lambda: 3–8 s; Fargate: 30–60 s) |
| FAISS + BM25 artifacts fit entirely in memory (<10 MB) | Index updates require re-building and re-uploading artifacts |
| No managed service lock-in | No built-in dashboard or monitoring (must add CloudWatch) |
| Can run locally with identical code (same FAISS/BM25) | Need to handle artifact versioning and cache invalidation |

---

## 4. Option B — Aurora PostgreSQL Serverless v2 + pgvector

### How It Works

1. **Storage**: chunks stored in a PostgreSQL table with a `vector(1024)` column
   (pgvector) and a `tsvector` column (full-text search).
2. **Hybrid search**: a single SQL query combines `<=>` (cosine distance) for
   vector search with `ts_rank()` for BM25-like lexical scoring, then fuses
   results (e.g., weighted sum or RRF in SQL).
3. **Infrastructure**: Aurora Serverless v2 scales ACUs based on demand and can
   idle at the minimum (0.5 ACU).

### Cost Estimate

| Component | Monthly cost |
|-----------|-------------|
| Aurora Serverless v2 — 0.5 ACU minimum × 730 hrs × US$0.12/ACU-hr | ~US$43.80 |
| Storage (< 100 MB) | < US$0.10 |
| I/O requests (~1k/month) | < US$0.20 |
| **Total** | **~US$44/month** |

> **Note**: Aurora Serverless v2 cannot scale to zero ACU — the minimum is 0.5 ACU.
> For a truly idle workload this still represents a fixed monthly floor of ~US$44.

### Hybrid Search Quality

| Aspect | Assessment |
|--------|-----------|
| Vector search | pgvector supports L2, inner product, cosine with HNSW or IVFFlat indexes — comparable to OpenSearch |
| Lexical search | PostgreSQL `tsvector` + `ts_rank()` approximates BM25; `pg_trgm` adds trigram similarity; not identical to Lucene BM25 but close for short documents |
| Fusion | Implement in SQL (weighted sum, RRF via CTEs) or in application code |
| Reranking | Must be done in application code (no native support) |
| **Relevance parity** | ⚠️ Close but not identical — `ts_rank` is not true BM25; may need tuning |

### Engineering Effort

| Task | Estimate |
|------|----------|
| Aurora cluster provisioning (CDK / CloudFormation) | 1–2 days |
| Schema design + migrations (pgvector, tsvector, indexes) | 1 day |
| Implement `PostgresSearchService` (index, search, delete, stats) | 2–3 days |
| Hybrid search query (pgvector + tsvector fusion in SQL) | 1–2 days |
| Wire into FastAPI dependencies, routes, agent, MCP | 1 day |
| Tests (unit + integration with test database) | 1–2 days |
| **Total** | **7–11 days** |

### Pros / Cons

| Pros | Cons |
|------|------|
| Fully managed service with automatic backups | ~US$44/month floor even when idle |
| Standard SQL — familiar tooling, migrations, transactions | `ts_rank` is not true BM25 (may need `rank_bm25` extension or Bedrock KB) |
| Bedrock Knowledge Bases supports Aurora pgvector natively | Requires VPC configuration, security groups, IAM auth |
| Easy to add metadata filtering, joins, analytics | pgvector HNSW index rebuild on inserts can be slow for large batches |
| Scales well to 100k+ vectors with HNSW | More operational complexity than a serverless function |

---

## 5. Option C — Local FAISS + SQLite FTS5

### How It Works

1. **Storage**: FAISS index file + SQLite database (FTS5 virtual table) on
   local disk.
2. **Hybrid search**: load FAISS index with `faiss` library for vector search;
   query SQLite FTS5 for BM25 scoring; fuse results in Python.
3. **Use case**: local development, testing, CI pipelines.

### Cost Estimate

| Component | Monthly cost |
|-----------|-------------|
| Everything | **US$0.00** |

### Hybrid Search Quality

| Aspect | Assessment |
|--------|-----------|
| Vector search | FAISS — identical to Option A |
| Lexical search | SQLite FTS5 uses BM25 natively (`bm25()` function) — good quality |
| Fusion | Same application-level fusion as Option A |
| **Relevance parity** | ✅ Good for dev/test; same algorithms as Option A |

### Engineering Effort

| Task | Estimate |
|------|----------|
| Implement `LocalSearchService` (FAISS + SQLite FTS5) | 2–3 days |
| Build index script (chunk → embed → store locally) | 1 day |
| Wire into FastAPI as an alternative dependency | 0.5 day |
| Tests | 1 day |
| **Total** | **4–5 days** |

### Pros / Cons

| Pros | Cons |
|------|------|
| Zero cost | Not a production solution — no HA, no backups |
| Fast feedback loop (no network calls for search) | Data lives on local disk only |
| Great for CI/CD test pipelines | SQLite doesn't support concurrent writes well |
| Same FAISS code as Option A (shared implementation) | Must mock or skip embedding calls in offline mode |

---

## 6. Side-by-Side Comparison

| Criteria | Option A: Lambda/Fargate + FAISS + S3 | Option B: Aurora pgvector | Option C: Local FAISS + SQLite |
|----------|---------------------------------------|--------------------------|-------------------------------|
| **Monthly cost** | ~US$0.05 | ~US$44 | US$0.00 |
| **Cost vs. current** | **99.99% reduction** | **92% reduction** | N/A (dev only) |
| **Hybrid search quality** | ✅ True BM25 + FAISS cosine; full control | ⚠️ ts_rank ≈ BM25; pgvector cosine | ✅ BM25 (FTS5) + FAISS cosine |
| **Cold start** | Lambda: 3–8 s; Fargate: 30–60 s | Near-instant (always-on) | Instant (local) |
| **Engineering effort** | 7–11 days | 7–11 days | 4–5 days |
| **Operational complexity** | Low (S3 + Lambda/Fargate) | Medium (Aurora VPC, security groups, IAM) | None |
| **Scalability (10x–100x)** | Re-index + bigger Lambda memory; eventually needs a real DB | Scales naturally (ACU auto-scaling) | Not applicable |
| **Integration with stack** | Python, FastAPI, aioboto3 — fits perfectly | Needs `asyncpg` + SQL layer; different paradigm | Same Python libs as Option A |
| **Dev/test experience** | Run locally with same FAISS/BM25 code | Need local PostgreSQL + pgvector | ✅ Best — fully local, no AWS |
| **Index updates** | Rebuild + re-upload to S3 | SQL INSERT/UPDATE — instant | Rebuild local files |
| **Monitoring** | CloudWatch + custom metrics | Aurora CloudWatch metrics + Performance Insights | Logs only |

---

## 7. Recommendation

### Winner: Option A (Lambda/Fargate + FAISS + BM25 + S3)

**With Option C as the local development companion.**

**Rationale:**

1. **Cost**: At ~US$0.05/month, Option A is three orders of magnitude cheaper than
   the current ~US$536/month, and ~880× cheaper than Option B's ~US$44/month floor.
   For a project with dozens of queries per week, paying $44/month for a PostgreSQL
   cluster is still overpaying.

2. **Hybrid search quality**: Option A uses the same BM25 algorithm (via
   `rank_bm25`) and the same FAISS vector engine already configured in the current
   OpenSearch index. The fusion logic (weighted arithmetic mean or RRF) can be
   tuned with full control — no compromises on relevance.

3. **Architecture fit**: The current stack is Python + FastAPI + aioboto3. FAISS
   and `rank_bm25` are pure Python libraries that integrate naturally. The
   Fargate variant keeps the FastAPI app as-is, requiring no Lambda adapter.

4. **Dev/test story**: Option C (local FAISS + SQLite FTS5) shares the core
   search logic with Option A. Developers get instant local search with zero AWS
   cost, and CI pipelines can run search integration tests without provisioning
   infrastructure.

5. **Scalability path**: If usage grows 10×–100×, the FAISS index and BM25
   artifacts still fit in memory (even 200k chunks at 1024 dims ≈ 800 MB). Beyond
   that, migrating to Aurora pgvector (Option B) becomes justified because the
   query volume would amortize the fixed cost.

### Suggested Variant: Fargate Spot with Scale-to-Zero

Use **ECS Fargate Spot** with scale-to-zero auto-scaling. This keeps the existing
FastAPI application running without modification, avoids Lambda cold starts and
packaging complexity, and costs essentially nothing when idle.

The index pipeline runs as a one-time ECS task or local script that uploads FAISS +
BM25 artifacts to S3. The FastAPI app loads them on startup (or lazily on first
query).

---

## 8. Implementation Plan

### Phase 1 — Local Search Engine (Option C) — ~5 days

1. Add `faiss-cpu`, `rank-bm25`, and `numpy` to `environment.yml`
2. Create `app/services/local_search_service.py` implementing the same interface
   as `OpenSearchService` (search, index, delete, stats)
3. Build an offline index script: `scripts/build_local_index.py` (chunk → embed →
   FAISS index + BM25 corpus → save to disk)
4. Wire into FastAPI as a configurable backend (env var `SEARCH_BACKEND=local|opensearch`);
   add `SEARCH_BACKEND` to `.env.example` with documentation
5. Add unit tests with the fake client pattern

### Phase 2 — S3-Backed Search (Option A) — ~5 days

1. Create `app/services/s3_search_service.py` — loads artifacts from S3, runs
   hybrid search
2. Build index pipeline: `scripts/build_and_upload_index.py` (extends Phase 1
   script to upload to S3)
3. Add Fargate task definition (or Lambda packaging) for production deployment
4. Update shared tools, agent, and MCP wrappers to use the new backend
5. Integration tests with mocked S3 (moto or fake)

### Phase 3 — Migrate and Deprecate OpenSearch — ~2 days

1. Run relevance comparison (same queries, compare top-K results vs. OpenSearch)
2. Switch production to Option A backend
3. Remove OpenSearch Serverless collection (save ~US$536/month)
4. Update documentation and `.env.example`

---

## 9. Migration Impact on Codebase

The key design principle is to keep the **service interface unchanged** so that
routes, agent tools, and MCP wrappers require minimal or no changes.

| Layer | Change required |
|-------|----------------|
| `app/models/` | None — `SearchResponse`, `SearchHit` etc. stay the same |
| `app/routes/opensearch.py` | Rename to `app/routes/search.py`; swap dependency injection |
| `app/services/dependencies.py` | Add factory for new search service; route based on env var |
| `shared/opensearch_tools.py` | Rename to `shared/search_tools.py`; point to new service |
| `agent/agent_factory.py` | Update agent name/description; tools auto-follow |
| `mcp_server/opensearch_tools.py` | Rename; point to new service |
| `tests/` | Add new test files; keep existing as regression until cutover |

---

## 10. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| FAISS index corruption on S3 | Versioned S3 bucket + checksum validation on download |
| Cold start latency (Fargate) | Pre-warm with CloudWatch scheduled event; or keep 1 task warm |
| BM25 quality differs from OpenSearch | Side-by-side relevance eval before cutover (Phase 3 step 1) |
| Embedding model change breaks index | Store model ID in index metadata; rebuild script validates |
| Concurrent index rebuilds | S3 object locking or atomic rename pattern |

---

## 11. References

- [OpenSearch Serverless Pricing](https://aws.amazon.com/opensearch-service/pricing/) — US$0.24/OCU-hr
- [Aurora Serverless v2 Pricing](https://aws.amazon.com/rds/aurora/pricing/) — US$0.12/ACU-hr
- [pgvector](https://github.com/pgvector/pgvector) — PostgreSQL vector extension
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) — Python BM25 implementation
- [SQLite FTS5](https://www.sqlite.org/fts5.html) — Full-text search extension
- Current index config: `docs/opensearch_index_setup.md`
- Current hybrid search weights: 0.3 BM25 + 0.7 neural (arithmetic mean normalization)
