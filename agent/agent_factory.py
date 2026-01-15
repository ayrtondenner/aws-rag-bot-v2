from __future__ import annotations

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from .settings import Settings
from .tools import build_document_tools, build_s3_tools


def build_root_agent(settings: Settings) -> Agent:
    """Create the root ADK Agent instance.

    Kept as a factory so importing modules doesn't create side effects.
    """
    s3_agent = build_s3_agent(settings)
    document_agent = build_document_agent(settings)
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
            "- document_agent: Local documentation operations (list files in the local sagemaker-docs folder).\n"
        ),
        sub_agents=[s3_agent, document_agent],
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
        description="Agent for local documentation operations (list local sagemaker-docs files).",
        instruction=(
            "You are the document agent. You help the user work with local documentation files in the repository. "
            "Use your tools to list the filenames available in the local sagemaker-docs folder. "
            "If the user asks for non-document tasks, do NOT attempt to solve them here; "
            "call the tool `document_transfer_to_root` to transfer control back to the root agent."
        ),
        tools=build_document_tools(),
    )