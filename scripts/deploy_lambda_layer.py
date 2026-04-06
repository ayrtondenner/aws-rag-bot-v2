"""Deploy Lambda layer with Python dependencies (faiss-cpu, rank-bm25, numpy).

Usage:
    conda run --prefix .venv python scripts/deploy_lambda_layer.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
LAMBDA_FUNCTION_NAME = os.getenv("SEARCH_LAMBDA_FUNCTION_NAME", "ragbot-search")
LAYER_NAME = "ragbot-search-deps"

# Path to the pre-built layer zip (built via pip install --platform manylinux2014_x86_64)
LAYER_ZIP = Path(os.environ.get("LAYER_ZIP", r"C:\Users\Ayrton\AppData\Local\Temp\lambda-layer.zip"))


def main() -> None:
    if not LAYER_ZIP.exists():
        print(f"Layer zip not found: {LAYER_ZIP}")
        sys.exit(1)

    zip_bytes = LAYER_ZIP.read_bytes()
    print(f"Layer zip size: {len(zip_bytes) / 1024 / 1024:.1f} MB")

    client = boto3.client("lambda", region_name=AWS_REGION)

    # 1. Publish the layer
    print(f"Publishing layer '{LAYER_NAME}'...")
    layer_resp = client.publish_layer_version(
        LayerName=LAYER_NAME,
        CompatibleRuntimes=["python3.11"],
        Content={"ZipFile": zip_bytes},
        Description="faiss-cpu, rank-bm25, numpy for ragbot-search Lambda",
    )
    layer_arn = layer_resp["LayerVersionArn"]
    print(f"Layer published: {layer_arn}")

    # 2. Attach the layer to the Lambda function
    print(f"Attaching layer to '{LAMBDA_FUNCTION_NAME}'...")
    config_resp = client.update_function_configuration(
        FunctionName=LAMBDA_FUNCTION_NAME,
        Layers=[layer_arn],
    )
    print(f"Lambda updated. Layers: {config_resp.get('Layers', [])}")
    print("Done!")


if __name__ == "__main__":
    main()
