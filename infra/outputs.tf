output "lambda_function_name" {
  description = "Name of the search Lambda function"
  value       = aws_lambda_function.search.function_name
}

output "lambda_function_arn" {
  description = "ARN of the search Lambda function"
  value       = aws_lambda_function.search.arn
}

output "s3_bucket_name" {
  description = "S3 bucket for search index artifacts"
  value       = aws_s3_bucket.index.id
}

output "iam_role_arn" {
  description = "IAM execution role ARN for the Lambda function"
  value       = aws_iam_role.lambda_execution.arn
}
