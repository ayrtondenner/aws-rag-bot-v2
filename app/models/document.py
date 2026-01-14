from __future__ import annotations

from pydantic import BaseModel, Field


class SplitTextRequest(BaseModel):
    text: str = Field(..., description="Input text to split")


class SplitTextResponse(BaseModel):
    count: int
    chunk_size: int
    chunk_overlap: int
    chunks: list[str]
