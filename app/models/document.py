from __future__ import annotations

from pydantic import BaseModel, Field


class SplitTextRequest(BaseModel):
    text: str = Field(..., description="Input text to split")


class SplitTextResponse(BaseModel):
    count: int
    chunk_size: int
    chunk_overlap: int
    chunks: list[str]


class EmbedTextRequest(BaseModel):
    text: str = Field(..., description="Input text to embed")


class EmbedTextResponse(BaseModel):
    dimensions: int
    embedding: list[float]


class LocalDocumentsResponse(BaseModel):
    count: int
    documents: list[str]


class LocalDocumentContentResponse(BaseModel):
    filename: str
    content: str
