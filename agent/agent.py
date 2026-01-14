from __future__ import annotations

from .agent_factory import build_root_agent
from .settings import Settings

settings = Settings()
settings.validate()

# ADK CLI (`adk run`, `adk web`) looks for a top-level `root_agent` (or `app`) symbol.
root_agent = build_root_agent(settings)