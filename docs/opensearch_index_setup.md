# OpenSearch Index Setup — Dashboard Dev Tools

> **See also**: The [RAG and OpenSearch](https://github.com/ayrtondenner/aws-rag-bot-v2/wiki/RAG-and-OpenSearch#opensearch-index-setup) wiki page contains these same instructions alongside additional context on the RAG pipeline, hybrid search, and index mappings.

> **One-time setup commands** to run in the **OpenSearch Dashboard → Dev Tools** console.
>
> These create the ML connector (Bedrock Titan Embed v2), register & deploy the model,
> create the ingest + search pipelines, and finally create the `sagemaker-docs` index
> with the correct mappings.

---

## 1. Create the ML Connector (Bedrock Titan Embed v2)

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

## 2. Register the Model

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

## 3. Deploy the Model

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

## 4. Create the Ingest Pipeline

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

## 5. Create the Search Pipeline

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

## 6. Create the Index

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

## 7. Verify

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
