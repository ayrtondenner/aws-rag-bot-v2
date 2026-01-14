
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.models.document import SplitTextRequest, SplitTextResponse
from app.services.dependencies import get_document_service
from app.services.document_service import DocumentService

router = APIRouter(prefix="/document", tags=["document"])


@router.post("/chunks", response_model=SplitTextResponse)
async def chunk_text(
	payload: SplitTextRequest,
	chunk_size: int = Query(default=500, ge=1),
	chunk_overlap: int = Query(default=50, ge=0),
	documents: DocumentService = Depends(get_document_service),
) -> SplitTextResponse:
	chunks = documents.chunk_text(text=payload.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	return SplitTextResponse(count=len(chunks), chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunks=chunks)

