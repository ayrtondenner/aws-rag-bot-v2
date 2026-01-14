from __future__ import annotations

from pydantic import BaseModel, Field


class SplitTextRequest(BaseModel):
    text: str = Field(..., description="Input text to split")


class SplitTextResponse(BaseModel):
    count: int
    chunks: list[str]


class EmbedTextRequest(BaseModel):
    text: str = Field(..., description="Input text to embed")


class EmbedTextResponse(BaseModel):
    dimensions: int
    embedding: list[float]
