# aws-rag-bot-v2

RAG (Retrieval-Augmented Generation) backend that ingests documents from AWS S3 and serves a FastAPI API for querying them. The goal is to evolve this into a production-ready, hybrid-search RAG service on AWS (S3 + OpenSearch) with solid testing, docs, and an agent layer.

## Functionality checklist

- [x] S3 integration
- [ ] Hybrid search using OpenSearch
- [ ] FastAPI unit tests
- [ ] Swagger documentation
- [ ] Google ADK agent

## S3 API routes

Base path: `/s3`

| Method | Path | Description | Query params |
| --- | --- | --- | --- |
| GET | `/s3/bucket/exists` | Check whether the configured S3 bucket exists (and is accessible). | _None_ |
| GET | `/s3/bucket/files/count` | Count objects in the configured S3 bucket. | `prefix` (optional) |
| GET | `/s3/files` | List objects in the configured S3 bucket. | `prefix` (optional) |
| GET | `/s3/file/content` | Get raw content of an object by key. | `file_name` (required) |
