from __future__ import annotations

import os
import uuid
from dataclasses import dataclass


DEFAULT_BEDROCK_INFERENCE_PROFILE_ID = "global.anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-sonnet-4-20250514-v1:0"

@dataclass(frozen=True)
class Settings:
    app_name: str = "aws_rag_bot"
    session_id: str = str(uuid.uuid4())
    user_id: str = "1"
    user_name: str = "Ayrton Denner"
    user_email: str = "ayrtondenner_2013@hotmail.com"

    _anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Amazon Bedrock (Claude) configuration.
    # Preferred: use a system-defined inference profile ID (cross-region) for reliability.
    bedrock_inference_profile_id: str = os.getenv(
        "BEDROCK_INFERENCE_PROFILE_ID", DEFAULT_BEDROCK_INFERENCE_PROFILE_ID
    ).strip()
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID).strip()

    # The actual LiteLLM model string used by ADK.
    # If ANTHROPIC_MODEL is set, we use it verbatim; otherwise we default to Bedrock inference profile.
    _anthropic_model: str = (
        os.getenv("ANTHROPIC_MODEL", "").strip()
        or f"bedrock/{os.getenv('BEDROCK_INFERENCE_PROFILE_ID', DEFAULT_BEDROCK_INFERENCE_PROFILE_ID).strip()}"
    )

    def validate(self) -> None:
        # ANTHROPIC_API_KEY may or may not be required depending on your LiteLLM setup.
        if not self._anthropic_model:
            raise ValueError(
                "Missing model configuration. Set ANTHROPIC_MODEL (e.g., ANTHROPIC_MODEL=bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0) "
                "or set BEDROCK_INFERENCE_PROFILE_ID/BEDROCK_MODEL_ID."
            )

        if not self.bedrock_inference_profile_id:
            raise ValueError(
                f"BEDROCK_INFERENCE_PROFILE_ID is not set (e.g., {DEFAULT_BEDROCK_INFERENCE_PROFILE_ID})."
            )

        if not self.bedrock_model_id:
            raise ValueError(f"BEDROCK_MODEL_ID is not set (e.g., {DEFAULT_BEDROCK_MODEL_ID}).")
