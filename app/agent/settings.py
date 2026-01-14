from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    app_name: str = "aws_rag_bot"
    session_id: str = str(uuid.uuid4())
    user_id: str = "1"
    user_name: str = "Ayrton Denner"
    user_email: str = "ayrtondenner_2013@hotmail.com"

    _anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    _anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4")

    def validate(self) -> None:
        # ANTHROPIC_API_KEY may or may not be required depending on your LiteLLM setup.
        if not self._anthropic_model:
            raise ValueError(
                "ANTHROPIC_MODEL is not set. Add it to your environment or .env (e.g., ANTHROPIC_MODEL=claude-sonnet-4)."
            )
