from __future__ import annotations
from .tools import mcp

def main() -> None:
	mcp.settings.port = 8002
	mcp.run(transport="streamable-http")

if __name__ == "__main__":
	main()
