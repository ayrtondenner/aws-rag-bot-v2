
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette import status

from app.models.document import EmbedTextRequest, EmbedTextResponse, SplitTextRequest, SplitTextResponse
from app.services.dependencies import get_document_service
from app.services.document_service import DocumentService, DocumentServiceError

router = APIRouter(prefix="/document", tags=["document"])


@router.post("/chunks", response_model=SplitTextResponse)
async def chunk_text(
	payload: SplitTextRequest,
	chunk_size: int = Query(default=500, ge=1),
	chunk_overlap: int = Query(default=50, ge=0),
	documents: DocumentService = Depends(get_document_service),
) -> SplitTextResponse:
	try:
		chunks = documents.chunk_text(text=payload.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	except ValueError as exc:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

	return SplitTextResponse(count=len(chunks), chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunks=chunks)


@router.post("/embed", response_model=EmbedTextResponse)
async def embed_text(
	payload: EmbedTextRequest,
	documents: DocumentService = Depends(get_document_service),
) -> EmbedTextResponse:
	try:
		embedding = documents.embed_text(text=payload.text)
	except ValueError as exc:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
	except DocumentServiceError as exc:
		raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

	return EmbedTextResponse(dimensions=len(embedding), embedding=embedding)

