# Placeholder zip — the real code is deployed via scripts/build_search_index.py --deploy-lambda
# or the SearchSetupService at app startup.
data "archive_file" "lambda_placeholder" {
  type        = "zip"
  output_path = "${path.module}/.terraform/lambda_placeholder.zip"

  source {
    content  = "def lambda_handler(event, context): return {'statusCode': 200}"
    filename = "handler.py"
  }
}

resource "aws_lambda_function" "search" {
  function_name = var.lambda_function_name
  role          = aws_iam_role.lambda_execution.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.11"
  timeout       = var.lambda_timeout
  memory_size   = var.lambda_memory_size

  filename         = data.archive_file.lambda_placeholder.output_path
  source_code_hash = data.archive_file.lambda_placeholder.output_base64sha256

  reserved_concurrent_executions = var.lambda_reserved_concurrency

  environment {
    variables = {
      SEARCH_INDEX_BUCKET        = var.index_bucket_name
      SEARCH_INDEX_PREFIX        = var.index_prefix
      BEDROCK_EMBEDDING_MODEL_ID = var.embedding_model_id
      BEDROCK_EMBEDDING_DIM      = tostring(var.embedding_dim)
      BM25_WEIGHT                = var.bm25_weight
      VECTOR_WEIGHT              = var.vector_weight
    }
  }

  # Ignore code changes — real code is deployed outside Terraform
  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.lambda_function_name}"
  retention_in_days = 14
}
