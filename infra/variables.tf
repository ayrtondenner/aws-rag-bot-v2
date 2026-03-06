variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used as prefix for resource naming"
  type        = string
  default     = "ragbot"
}

variable "index_bucket_name" {
  description = "S3 bucket name for FAISS/BM25 index artifacts"
  type        = string
}

variable "index_prefix" {
  description = "S3 key prefix for index artifacts"
  type        = string
  default     = "search-index/"
}

variable "lambda_function_name" {
  description = "Name of the search Lambda function"
  type        = string
  default     = "ragbot-search"
}

variable "lambda_memory_size" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 512
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 30
}

variable "lambda_reserved_concurrency" {
  description = "Reserved concurrent executions for the Lambda function"
  type        = number
  default     = 5
}

variable "embedding_model_id" {
  description = "Bedrock embedding model ID"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "embedding_dim" {
  description = "Embedding vector dimension"
  type        = number
  default     = 1024
}

variable "bm25_weight" {
  description = "BM25 score weight in hybrid fusion"
  type        = string
  default     = "0.3"
}

variable "vector_weight" {
  description = "Vector score weight in hybrid fusion"
  type        = string
  default     = "0.7"
}
