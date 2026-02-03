from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from typing import Any, Optional

import anyio

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


EXPECTED_TOOL_NAMES = {
    "s3_bucket_exists",
    "s3_list_bucket_files",
    "s3_get_file_content",
    "list_local_sagemaker_docs",
}


def _wait_for_port(host: str, port: int, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except Exception as exc:  # pragma: no cover
            last_err = exc
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {host}:{port} ({last_err})")


async def _run_client_checks(url: str, bucket_name: Optional[str]) -> None:
    async with streamable_http_client(url) as (read_stream, write_stream, _get_session_id):
        session = ClientSession(read_stream, write_stream)
        await session.initialize()

        tools = await session.list_tools()
        tool_names = {t.name for t in tools.tools}

        missing = EXPECTED_TOOL_NAMES - tool_names
        if missing:
            raise RuntimeError(f"Server is missing expected tools: {sorted(missing)}")

        # Always run a tool that doesn't require AWS credentials.
        docs_result = await session.call_tool("list_local_sagemaker_docs")
        if docs_result.isError:
            raise RuntimeError(f"list_local_sagemaker_docs returned error: {docs_result}")

        # Optionally run an S3 call if bucket is available.
        effective_bucket = (bucket_name or "").strip() or (os.getenv("S3_BUCKET_NAME") or "").strip() or None
        if effective_bucket:
            s3_result = await session.call_tool(
                "s3_bucket_exists",
                arguments={"bucket_name": effective_bucket},
            )
            if s3_result.isError:
                raise RuntimeError(f"s3_bucket_exists returned error: {s3_result}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for aws-rag-bot MCP server")

    parser.add_argument(
        "--url",
        default=os.getenv("MCP_URL", "http://127.0.0.1:8001/mcp"),
        help="Full MCP Streamable HTTP endpoint URL (default: http://127.0.0.1:8001/mcp)",
    )
    parser.add_argument(
        "--bucket-name",
        default=os.getenv("S3_BUCKET_NAME"),
        help="Optional bucket name to validate s3_bucket_exists.",
    )

    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start the MCP server in a subprocess for the duration of the test.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # If we're starting the server ourselves and the user didn't override URL,
    # align it with the selected host/port.
    default_url = "http://127.0.0.1:8001/mcp"
    if args.start_server and args.url == default_url:
        args.url = f"http://{args.host}:{args.port}/mcp"

    proc: subprocess.Popen[str] | None = None
    try:
        if args.start_server:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mcp_server.main",
                    "--transport",
                    "streamable-http",
                    "--host",
                    str(args.host),
                    "--port",
                    str(args.port),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _wait_for_port(args.host, args.port, timeout_s=15.0)

        anyio.run(_run_client_checks, args.url, args.bucket_name)
        print("OK: MCP smoke test passed")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
