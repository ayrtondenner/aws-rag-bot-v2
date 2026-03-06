from __future__ import annotations

from mcp_server import mcp

# Side-effect imports: register @mcp.resource decorators on the shared instance.
import mcp_server.s3_tools  # noqa: F401
import mcp_server.document_tools  # noqa: F401
import mcp_server.search_tools  # noqa: F401


def main() -> None:
	mcp.settings.port = 8002
	mcp.run(transport="streamable-http")

if __name__ == "__main__":
	main()
