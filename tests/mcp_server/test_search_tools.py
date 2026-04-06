"""Tests for MCP search tool/resource classification.

Verifies that write operations are registered as @mcp.tool and
read-only operations remain as @mcp.resource on the FastMCP instance.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from mcp_server import mcp

# Side-effect import to trigger registration of tools/resources.
import mcp_server.search_tools  # noqa: F401


# ---------------------------------------------------------------------------
# Tool / Resource classification
# ---------------------------------------------------------------------------


def _get_tool_names() -> set[str]:
    tools = asyncio.run(mcp.get_tools())
    return set(tools.keys())


def _get_resource_names() -> set[str]:
    """Get all resource names (static + templates)."""

    # Static resources are keyed by URI
    resources = asyncio.run(mcp.get_resources())
    names = set()
    for r in resources.values():
        names.add(getattr(r, "name", None) or str(r))

    # Parameterized resources are templates, keyed by URI template
    templates = asyncio.run(mcp.get_resource_templates())
    for t in templates.values():
        names.add(getattr(t, "name", None) or str(t))

    return names


def test_search_index_document_is_tool():
    """search_index_document modifies the index — must be a tool, not resource."""

    assert "search_index_document" in _get_tool_names()
    assert "search_index_document" not in _get_resource_names()


def test_search_delete_document_is_tool():
    """search_delete_document modifies the index — must be a tool."""

    assert "search_delete_document" in _get_tool_names()


def test_search_delete_by_filename_is_tool():
    """search_delete_by_filename modifies the index — must be a tool."""

    assert "search_delete_by_filename" in _get_tool_names()


def test_read_only_functions_are_resources():
    """Read-only operations must remain as resources, not tools."""

    resource_names = _get_resource_names()
    tool_names = _get_tool_names()

    expected_resources = {
        "search_query",
        "search_document_exists",
        "search_list_indexed_documents",
        "search_get_index_stats",
    }

    for name in expected_resources:
        assert name in resource_names, f"{name} should be a resource"
        assert name not in tool_names, f"{name} should NOT be a tool"


def test_write_operations_not_in_resources():
    """Write operations must not appear in the resource list."""

    resource_names = _get_resource_names()

    write_ops = {"search_index_document", "search_delete_document", "search_delete_by_filename"}
    for name in write_ops:
        assert name not in resource_names, f"{name} is a write op and should NOT be a resource"


# ---------------------------------------------------------------------------
# Shared tool wrappers — delete functions
# ---------------------------------------------------------------------------


@patch("shared.search_tools._get_search_service")
def test_shared_delete_document_calls_service(mock_get_svc):
    mock_svc = AsyncMock()
    mock_svc.delete_document.return_value = True
    mock_get_svc.return_value = mock_svc

    from shared.search_tools import search_delete_document

    result = asyncio.run(search_delete_document(doc_id="abc-123"))

    assert result == {"deleted": True}
    mock_svc.delete_document.assert_awaited_once_with(doc_id="abc-123")


@patch("shared.search_tools._get_search_service")
def test_shared_delete_by_filename_calls_service(mock_get_svc):
    mock_svc = AsyncMock()
    mock_svc.delete_documents_by_filename.return_value = 5
    mock_get_svc.return_value = mock_svc

    from shared.search_tools import search_delete_by_filename

    result = asyncio.run(search_delete_by_filename(filename="old-doc.md"))

    assert result == {"deleted_count": 5}
    mock_svc.delete_documents_by_filename.assert_awaited_once_with(filename="old-doc.md")
