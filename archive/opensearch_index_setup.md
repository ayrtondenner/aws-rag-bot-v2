# OpenSearch Serverless — Complete Setup Guide

> **See also**: The [RAG and OpenSearch](https://github.com/ayrtondenner/aws-rag-bot-v2/wiki/RAG-and-OpenSearch#opensearch-index-setup) wiki page contains additional context on the RAG pipeline, hybrid search, and index mappings.

This guide covers the full infrastructure setup needed to run the RAG Bot v2 OpenSearch pipeline — from IAM roles and AOSS policies to pipelines and the vector index.

---

## Prerequisites Checklist

Before creating the OpenSearch collection and index, the following must be in place:

1. **IAM connector role** (`opensearch-bedrock-connector-role`)
   - Permission policy: `bedrock:InvokeModel` on `amazon.titan-embed-text-v2:0`
   - Trust policy: allows `aoss.amazonaws.com` to assume the role
2. **AOSS encryption policy** — AWS-owned key for the collection
3. **AOSS network policy** — public access for dashboards and API
4. **AOSS data access policy** — must include ALL principals that will touch the collection:
   - The caller identity (check with `aws sts get-caller-identity`)
   - The connector role ARN
   - Any additional IAM users or roles
   - Collection, index, and model permissions (this was the main pain point — see `docs/opensearch-aoss-permissions-summary.md`)
5. **AOSS collection** (`ragbot-v2-collection`, type: VECTORSEARCH) — must be in ACTIVE state

> **Key gotcha**: The principal you're logged in with must be explicitly listed in the data access policy. It's not automatic — even admin users get "access denied" if their ARN isn't included.

---

## Automated Setup (Recommended)

The `scripts/setup_opensearch.py` script automates the entire setup end-to-end:

- **Phase 1**: Creates the IAM connector role with Bedrock permissions
- **Phase 2**: Creates AOSS encryption, network, and data access policies + the collection
- **Phase 3**: Creates the ML connector, model, pipelines, and index via the OpenSearch API
- **Verification**: Indexes a test document and confirms the embedding was generated

### Running the Script

```bash
conda run --prefix .venv python scripts/setup_opensearch.py
```

### Configuration

The script uses sensible defaults matching the project. Override via environment variables:

| Value | Default | Env Var |
|-------|---------|---------|
| Region | `us-west-2` | `AWS_REGION` |
| Collection name | `ragbot-v2-collection` | `OPENSEARCH_COLLECTION_NAME` |
| Index name | `sagemaker-docs` | `OPENSEARCH_INDEX_NAME` |
| Embedding dimension | `1024` | `BEDROCK_EMBEDDING_DIM` |
| Extra principals | none | `AOSS_ADDITIONAL_PRINCIPALS` (comma-separated ARNs) |

The caller identity is auto-detected via `sts:GetCallerIdentity`.

### Idempotency

The script is safe to re-run:
- Existing IAM roles, AOSS policies, and collections are detected and skipped
- Existing ML connectors and models are found by name and reused
- Existing pipelines and indexes are detected and skipped
- The data access policy is updated (not duplicated) if it already exists

### Output

On completion, the script prints the `OPENSEARCH_ENDPOINT` to add to your `.env` file.

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Access denied" on Phase 3 OpenSearch calls | Data access policy not propagated yet | Wait 30–60 seconds and re-run (the script includes a built-in delay) |
| "Collection not found" | Collection deleted or wrong name | Check `OPENSEARCH_COLLECTION_NAME` |
| "Model deployment failed" | Connector role trust policy incorrect | Verify the role's trust policy allows `aoss.amazonaws.com` |
| Embedding not populated in verification | Ingest pipeline or model not working | Check model state in Dashboard Dev Tools: `GET /_plugins/_ml/models/<MODEL_ID>` |

---

## Manual Setup (Dashboard Dev Tools)

If you prefer to run the steps manually in **OpenSearch Dashboard → Dev Tools**, use the commands below. These correspond to Phase 3 of the automated script (the IAM and AOSS prerequisites from Phases 1–2 must already be in place).

### 1. Create the ML Connector (Bedrock Titan Embed v2)

```json
POST /_plugins/_ml/connectors/_create
{
  "name": "Amazon Bedrock - Titan Embed v2",
  "description": "Connector for Amazon Titan Text Embeddings V2 via Bedrock",
  "version": 1,
  "protocol": "aws_sigv4",
  "parameters": {
    "region": "us-west-2",
    "service_name": "bedrock"
  },
  "credential": {
    "roleArn": "arn:aws:iam::916475406025:role/opensearch-bedrock-connector-role"
  },
  "actions": [
    {
      "action_type": "predict",
      "method": "POST",
      "url": "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v2:0/invoke",
      "headers": {
        "content-type": "application/json",
        "x-amz-content-sha256": "required"
      },
      "request_body": "{ \"inputText\": \"${parameters.inputText}\" }",
      "pre_process_function": "\n    StringBuilder builder = new StringBuilder();\n    builder.append(\"{\" );\n    builder.append(\"\\\"parameters\\\":{\" );\n    builder.append(\"\\\"inputText\\\":\\\"\" );\n    builder.append(params.text_docs[0]);\n    builder.append(\"\\\"\" );\n    builder.append(\"}\" );\n    builder.append(\"}\" );\n    def result = builder.toString();\n    return result;\n  ",
      "post_process_function": "\n    def name = \"sentence_embedding\";\n    def dataType = \"FLOAT32\";\n    if (params.embedding != null) {\n      def shape = [params.embedding.length];\n      def json = \"{\" +\n        \"\\\"name\\\":\\\"\" + name + \"\\\",\" +\n        \"\\\"data_type\\\":\\\"\" + dataType + \"\\\",\" +\n        \"\\\"shape\\\":\" + shape + \",\" +\n        \"\\\"data\\\":\" + params.embedding +\n        \"}\";\n      return json;\n    }\n    return \"{\\\"error\\\":\\\"No embedding returned\\\"}\";\n  "
    }
  ]
}
```

> **Save the `connector_id`** from the response — you'll need it in step 2.

---

### 2. Register the Model

Replace `<CONNECTOR_ID>` with the value from step 1.

```json
POST /_plugins/_ml/models/_register
{
  "name": "Bedrock Titan Embed v2",
  "function_name": "remote",
  "description": "Titan Text Embedding V2 via Bedrock connector",
  "connector_id": "<CONNECTOR_ID>"
}
```

> **Save the `model_id`** from the response.

---

### 3. Deploy the Model

Replace `<MODEL_ID>` with the value from step 2.

```json
POST /_plugins/_ml/models/<MODEL_ID>/_deploy
```

Wait a few seconds, then verify it's deployed:

```json
GET /_plugins/_ml/models/<MODEL_ID>
```

You should see `"model_state": "DEPLOYED"`.

---

### 4. Create the Ingest Pipeline

Replace `<MODEL_ID>` with the value from step 2.

```json
PUT /_ingest/pipeline/sagemaker-docs-ingest-pipeline
{
  "description": "Embeds the 'content' field using Titan Embed v2",
  "processors": [
    {
      "text_embedding": {
        "model_id": "<MODEL_ID>",
        "field_map": {
          "content": "content_embedding"
        }
      }
    }
  ]
}
```

---

### 5. Create the Search Pipeline

```json
PUT /_search/pipeline/sagemaker-docs-search-pipeline
{
  "description": "Normalisation + weighted combination for hybrid search",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "min_max"
        },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.3, 0.7]
          }
        }
      }
    }
  ]
}
```

> Weights: `0.3` for BM25 (text), `0.7` for neural (vector). Adjust as needed.

---

### 6. Create the Index

```json
PUT /sagemaker-docs
{
  "settings": {
    "index": {
      "knn": true,
      "default_pipeline": "sagemaker-docs-ingest-pipeline"
    }
  },
  "mappings": {
    "properties": {
      "filename": {
        "type": "keyword"
      },
      "content": {
        "type": "text",
        "analyzer": "standard"
      },
      "content_embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "engine": "faiss",
          "space_type": "l2",
          "name": "hnsw",
          "parameters": {}
        }
      }
    }
  }
}
```

---

### 7. Verify

Test the ingest pipeline by indexing a single document:

```json
POST /sagemaker-docs/_doc
{
  "filename": "test-doc.md",
  "content": "Amazon SageMaker is a fully managed machine learning service."
}
```

Then retrieve it and confirm `content_embedding` was populated:

```json
GET /sagemaker-docs/_search
{
  "query": { "match_all": {} },
  "size": 1
}
```

Clean up the test document:

```json
POST /sagemaker-docs/_delete_by_query
{
  "query": { "term": { "filename": "test-doc.md" } }
}
```

---

## Notes

- The **connector IAM role** (`opensearch-bedrock-connector-role`) must have:
  - `bedrock:InvokeModel` permission on `arn:aws:bedrock:us-west-2::foundation-model/amazon.titan-embed-text-v2:0`
  - A trust policy allowing `aoss.amazonaws.com` to assume it.
- Pipeline names (`sagemaker-docs-ingest-pipeline`, `sagemaker-docs-search-pipeline`) must match the constants in `app/services/opensearch_service.py`.
- Index name (`sagemaker-docs`) must match `OPENSEARCH_INDEX_NAME` in `.env`.
- For a deep dive on IAM and AOSS permissions, see `docs/opensearch-aoss-permissions-summary.md`.
