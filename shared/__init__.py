from __future__ import annotations

from google.adk.tools.transfer_to_agent_tool import transfer_to_agent
from google.adk.tools.tool_context import ToolContext

# TODO: import this from S3 config instead
DEFAULT_SAGEMAKER_DOCS_BUCKET_NAME = "senior-sagemaker-assessment-bucket"


def transfer_to_root(tool_context: ToolContext) -> None:
    """Transfer control back to the root agent.

    Use this when the user's request is not about the sub-agent responsibility.
    """

    transfer_to_agent("root_agent", tool_context)
