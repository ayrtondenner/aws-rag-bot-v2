"""Phase 0.2-0.4 AWS Validation Script.

Validates FAISS+BM25 config, startup indexing readiness, and search functionality.
Run with: conda run --prefix .venv python scripts/aws_validation.py
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import boto3


def check_credentials():
    """Step 1: Verify AWS credentials."""
    print("=" * 60)
    print("STEP 1: AWS Credentials")
    print("=" * 60)
    sts = boto3.client("sts", region_name=os.getenv("AWS_REGION", "us-west-2"))
    identity = sts.get_caller_identity()
    print(f"  Account: {identity['Account']}")
    print(f"  ARN:     {identity['Arn']}")
    print("  Status:  OK")
    return True


def check_s3_bucket():
    """Step 2: Verify S3 bucket and search index artifacts."""
    print("\n" + "=" * 60)
    print("STEP 2: S3 Bucket & Search Index Artifacts (#6)")
    print("=" * 60)
    bucket = os.getenv("SEARCH_INDEX_BUCKET") or os.getenv("S3_BUCKET_NAME")
    prefix = os.getenv("SEARCH_INDEX_PREFIX", "search-index/")
    region = os.getenv("AWS_REGION", "us-west-2")

    s3 = boto3.client("s3", region_name=region)

    # Check bucket exists
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"  Bucket '{bucket}': EXISTS")
    except Exception as e:
        print(f"  Bucket '{bucket}': MISSING - {e}")
        return False

    # Check search index artifacts
    expected_artifacts = ["faiss.index", "corpus.pkl", "bm25.pkl"]
    all_found = True
    for artifact in expected_artifacts:
        key = f"{prefix}{artifact}"
        try:
            resp = s3.head_object(Bucket=bucket, Key=key)
            size = resp["ContentLength"]
            print(f"  Artifact '{key}': FOUND ({size:,} bytes)")
        except Exception:
            print(f"  Artifact '{key}': MISSING")
            all_found = False

    print(f"  Artifacts status: {'ALL PRESENT' if all_found else 'INCOMPLETE'}")
    return all_found


def check_lambda():
    """Step 3: Verify Lambda function exists and invoke with test payload."""
    print("\n" + "=" * 60)
    print("STEP 3: Lambda Function & Search Test (#6, #7)")
    print("=" * 60)
    function_name = os.getenv("SEARCH_LAMBDA_FUNCTION_NAME", "ragbot-search")
    region = os.getenv("AWS_REGION", "us-west-2")

    lambda_client = boto3.client("lambda", region_name=region)

    # Check function exists
    try:
        config = lambda_client.get_function_configuration(FunctionName=function_name)
        print(f"  Function '{function_name}': EXISTS")
        print(f"  Runtime:  {config['Runtime']}")
        print(f"  Memory:   {config['MemorySize']} MB")
        print(f"  Timeout:  {config['Timeout']}s")

        env_vars = config.get("Environment", {}).get("Variables", {})
        print(f"  Env vars: {list(env_vars.keys())}")

        embed_dim = env_vars.get("BEDROCK_EMBEDDING_DIM", "not set")
        print(f"  Embedding dim: {embed_dim}")
    except Exception as e:
        print(f"  Function '{function_name}': ERROR - {e}")
        return False

    # Test: get_stats
    print("\n  --- Invoking get_stats ---")
    try:
        resp = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps({"action": "get_stats"}),
        )
        payload = json.loads(resp["Payload"].read())
        if "error" in payload:
            print(f"  Stats error: {payload['error']}")
        else:
            print(f"  Stats: {json.dumps(payload, indent=4)}")
    except Exception as e:
        print(f"  Stats invocation failed: {e}")

    # Test: search
    print("\n  --- Invoking search (hybrid) ---")
    try:
        resp = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps({
                "action": "search",
                "query": "what is SageMaker",
                "size": 3,
                "search_type": "hybrid",
            }),
        )
        payload = json.loads(resp["Payload"].read())
        if "error" in payload:
            print(f"  Search error: {payload['error']}")
        else:
            hits = payload.get("hits", [])
            print(f"  Hits returned: {len(hits)}")
            for i, hit in enumerate(hits):
                print(f"    [{i+1}] score={hit.get('score', 'N/A'):.4f}  doc={hit.get('filename', 'unknown')}")
    except Exception as e:
        print(f"  Search invocation failed: {e}")

    # Test: list_documents
    print("\n  --- Invoking list_documents ---")
    try:
        resp = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps({"action": "list_documents"}),
        )
        payload = json.loads(resp["Payload"].read())
        if "error" in payload:
            print(f"  List error: {payload['error']}")
        else:
            filenames = payload.get("filenames", [])
            print(f"  Indexed documents: {len(filenames)}")
            if filenames:
                print(f"  Sample: {filenames[:5]}")
    except Exception as e:
        print(f"  List invocation failed: {e}")

    return True


def check_terraform():
    """Step 4: Check Terraform state consistency."""
    print("\n" + "=" * 60)
    print("STEP 4: Terraform State Check")
    print("=" * 60)
    state_file = os.path.join("infra", "terraform.tfstate")
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        resources = state.get("resources", [])
        print(f"  State file: FOUND")
        print(f"  Resources tracked: {len(resources)}")
        for r in resources:
            print(f"    - {r['type']}.{r['name']}")
    else:
        print("  State file: NOT FOUND (may be remote)")


def main():
    print("AWS RAG Bot v2 - Phase 0 Validation")
    print("=" * 60)
    print()

    try:
        check_credentials()
    except Exception as e:
        print(f"  FAILED: {e}")
        print("\nCannot proceed without valid credentials.")
        sys.exit(1)

    check_s3_bucket()
    check_lambda()
    check_terraform()

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
