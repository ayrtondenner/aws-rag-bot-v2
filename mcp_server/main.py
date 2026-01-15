from __future__ import annotations

import argparse
import os

from app.services.s3_service import S3ServiceError
from shared.tools import mcp


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="AWS RAG Bot MCP Server")
	parser.add_argument(
		"--transport",
		choices=["streamable-http", "sse", "stdio"],
		default="streamable-http",
		help="MCP transport to use.",
	)
	parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
	parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8001")))
	parser.add_argument(
		"--mount-path",
		default=os.getenv("MCP_MOUNT_PATH", "/"),
		help="ASGI mount path (mostly relevant for http transports).",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()

	# Configure host/port on the FastMCP instance (FastMCP.run doesn't accept them).
	mcp.settings.host = args.host
	mcp.settings.port = args.port

	try:
		mcp.run(transport=args.transport, mount_path=args.mount_path)
	except S3ServiceError as exc:
		# Keep error messages clean (avoid leaking stack traces into process logs unless desired).
		raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
	main()
