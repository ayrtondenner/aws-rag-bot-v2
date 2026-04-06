"""Quick test: can we call Bedrock Titan Embeddings?"""
import json
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

region = os.getenv("AWS_REGION", "us-west-2")
model = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

print(f"Region: {region}")
print(f"Model:  {model}")
print("Calling Bedrock...", flush=True)

try:
    client = boto3.client("bedrock-runtime", region_name=region)
    resp = client.invoke_model(
        modelId=model,
        body=json.dumps({"inputText": "hello world"}),
        contentType="application/json",
    )
    body = json.loads(resp["body"].read())
    dim = len(body["embedding"])
    print(f"SUCCESS: embedding dim={dim}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
