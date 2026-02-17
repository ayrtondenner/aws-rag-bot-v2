from __future__ import annotations

from typing import Annotated
from pydantic import Field

from app.models.document import LocalDocumentContentResponse, LocalDocumentsResponse
import shared.document_tools as shared_docs

from mcp_server import mcp


@mcp.resource(
    name="list_local_sagemaker_docs",
    description="List files in the local sagemaker-docs folder.",
    uri="local://sagemaker-docs/",
)
async def list_local_sagemaker_docs() -> LocalDocumentsResponse:
    """List files in the local `sagemaker-docs` folder.

    Returns:
        JSON with {count, documents:[filename,...]}.
    """

    return await shared_docs.list_local_sagemaker_docs()


@mcp.resource(
    name="get_local_sagemaker_doc_content",
    description="Get the text content of a local file in the sagemaker-docs folder by filename.",
    uri="local://sagemaker-docs/{filename}",
)
async def get_local_sagemaker_doc_content(
    *,
    filename: Annotated[
        str,
        Field(..., description="Filename of the local doc in the sagemaker-docs folder to read."),
    ],
) -> LocalDocumentContentResponse:
    """Read a local doc file content.

    Returns:
        JSON with {filename, content}.
    """

    return await shared_docs.get_local_sagemaker_doc_content(filename=filename)
