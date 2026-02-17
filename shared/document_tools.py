from __future__ import annotations

from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.llm_agent import ToolUnion

from app.models.document import LocalDocumentContentResponse, LocalDocumentsResponse
from app.services.dependencies import get_document_service as get_document_service_dependency
from app.services.document_service import DocumentService

from shared import transfer_to_root


def _get_document_service() -> DocumentService:
    """Build a DocumentService using the same factory path as the FastAPI dependency."""

    return get_document_service_dependency()


async def list_local_sagemaker_docs() -> LocalDocumentsResponse:
    """List files in the local `sagemaker-docs` folder.

    Returns:
        JSON with {count, documents:[filename,...]}.
    """

    documents = _get_document_service()
    result = documents.list_local_sagemaker_docs()
    return LocalDocumentsResponse.model_validate(result)


async def get_local_sagemaker_doc_content(*, filename: str) -> LocalDocumentContentResponse:
    """Read a local doc file content.

    Returns:
        JSON with {filename, content}.
    """

    documents = _get_document_service()
    content = documents.get_local_sagemaker_doc_content(filename=filename)
    return LocalDocumentContentResponse(filename=filename, content=content)


def build_document_tools() -> list[ToolUnion]:
    """Build the list of local-document ADK tools for the agent."""

    return [
        FunctionTool(list_local_sagemaker_docs),
        FunctionTool(get_local_sagemaker_doc_content),
        FunctionTool(transfer_to_root),
    ]
