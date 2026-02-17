from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP(
    name="aws-rag-bot-mcp",
    instructions=(
        "Shared tool server for AWS RAG Bot. "
        "Exposes S3, local-document and OpenSearch utilities designed to be "
        "reused by both ADK agents and the MCP server."
    ),
)
