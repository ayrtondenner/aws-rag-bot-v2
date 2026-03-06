from __future__ import annotations

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from .settings import Settings
from shared.s3_tools import build_s3_tools
from shared.document_tools import build_document_tools
from shared.search_tools import build_search_tools


def build_root_agent(settings: Settings) -> Agent:
    """Create the root ADK Agent instance.

    Kept as a factory so importing modules doesn't create side effects.
    """
    s3_agent = build_s3_agent(settings)
    document_agent = build_document_agent(settings)
    search_agent = build_search_agent(settings)
    return Agent(
        name="root_agent",
        model=LiteLlm(model=settings._anthropic_model),
        description="The root agent that delegates to specialized sub-agents.",
        instruction=(
            "You are the root agent. You are the main coordinator of the conversation. "
            "You are coordinating a team. Your task is to delegate user requests to the appropriate agent.\n\n"
            "You have specialized sub-agents:\n"
            "- s3_agent: S3 bucket operations (check existence, list files, fetch file content). "
            "If the user doesn't provide a bucket name, it defaults to the SageMaker docs bucket from env var S3_BUCKET_NAME.\n"
            "- document_agent: Local documentation operations (list files and fetch file content from the local sagemaker-docs folder).\n"
            "- search_agent: Hybrid search operations (index documents, search with BM25 + vector via FAISS, "
            "check document existence, list indexed documents, get index stats).\n"
        ),
        sub_agents=[s3_agent, document_agent, search_agent],
    )


def build_s3_agent(settings: Settings) -> Agent:
    return Agent(
        name="s3_agent",
        model=LiteLlm(model=settings._anthropic_model),
        description=(
            "Agent for S3 bucket operations (existence checks, listing, fetching content). "
            "If the user doesn't provide a bucket name, use the default SageMaker docs bucket."
        ),
        instruction=(
            "You are the S3 agent. You help the user interact with S3 buckets and objects. "
            "Use your tools to: check if a bucket exists, list files in a bucket, and fetch file contents. "
            "When a bucket name is required but the user doesn't provide one, it will use the default SageMaker docs bucket."

            "If the user asks for non-S3 tasks, do NOT attempt to solve them here; "
            "call the tool `s3_transfer_to_root` to transfer control back to the root agent."
        ),
        tools=build_s3_tools(),
    )


def build_document_agent(settings: Settings) -> Agent:
    return Agent(
        name="document_agent",
        model=LiteLlm(model=settings._anthropic_model),
        description="Agent for local documentation operations (list and read local sagemaker-docs files).",
        instruction=(
            "You are the document agent. You help the user work with local documentation files in the repository. "
            "Use your tools to list the filenames available in the local sagemaker-docs folder and fetch a file's content by filename. "
            "If the user asks for non-document tasks, do NOT attempt to solve them here; "
            "call the tool `document_transfer_to_root` to transfer control back to the root agent."
        ),
        tools=build_document_tools(),
    )


def build_search_agent(settings: Settings) -> Agent:
    """Build the search sub-agent for hybrid search operations.

    Args:
        settings: Agent settings (provides LLM model string).

    Returns:
        A configured search Agent.
    """

    return Agent(
        name="search_agent",
        model=LiteLlm(model=settings._anthropic_model),
        description=(
            "Agent for hybrid search operations (index documents, search with BM25 + vector via FAISS, "
            "check document existence, list indexed documents, get index stats)."
        ),
        instruction=(
            "You are the search agent. You help the user search and manage documents "
            "in the search index (FAISS + BM25 via Lambda).\n\n"
            "Your capabilities:\n"
            "- **Search**: Use `search_query` to find relevant documents. Default search type is 'hybrid' "
            "(combines BM25 text matching with FAISS vector search). You can also do 'text' or 'vector' only.\n"
            "- **Index**: Use `search_index_document` to add a document to the index (auto-chunks and skips duplicates).\n"
            "- **Check existence**: Use `search_document_exists` to see if a filename is already indexed.\n"
            "- **List documents**: Use `search_list_indexed_documents` to see all indexed filenames.\n"
            "- **Stats**: Use `search_get_index_stats` for index-level statistics.\n\n"
            "If the user asks for non-search tasks, call the `transfer_to_root` tool to hand back to the root agent."
        ),
        tools=build_search_tools(),
    )
