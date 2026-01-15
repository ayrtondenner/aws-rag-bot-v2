
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from starlette import status

from app.models.document import (
	EmbedTextRequest,
	EmbedTextResponse,
	LocalDocumentsResponse,
	SplitTextRequest,
	SplitTextResponse,
)
from app.services.dependencies import get_document_service
from app.services.document_service import DocumentService, DocumentServiceError

router = APIRouter(prefix="/document", tags=["document"])


@router.post(
	"/chunks",
	response_model=SplitTextResponse,
	responses={
		200: {
			"content": {
				"application/json": {
					"example": {
						"count": 3,
						"chunk_size": 500,
						"chunk_overlap": 50,
						"chunks": [
							"# Using the SageMaker Training and Inference Toolkits<a name=\"amazon-sagemaker-toolkits\"></a>",
							"The [SageMaker Training](https://github.com/aws/sagemaker-training-toolkit) and [SageMaker Inference](https://github.com/aws/sagemaker-inference-toolkit) toolkits implement the functionality that you need to adapt your containers to run scripts, train algorithms, and deploy models on SageMaker. When installed, the library defines the following for users:\n+ The locations for storing code and other resources.",
							"+ The entry point that contains the code to run when the container is started. Your Dockerfile must copy the code that needs to be run into the location expected by a container that is compatible with SageMaker. \n+ Other information that a container needs to manage deployments for training and inference.",
						],
					}
				}
			}
		}
	},
)
async def chunk_text(
	payload: SplitTextRequest = Body(
		...,
		examples=[{
			"text": "# Using the SageMaker Training and Inference Toolkits<a name=\"amazon-sagemaker-toolkits\"></a>\n\nThe [SageMaker Training](https://github.com/aws/sagemaker-training-toolkit) and [SageMaker Inference](https://github.com/aws/sagemaker-inference-toolkit) toolkits implement the functionality that you need to adapt your containers to run scripts, train algorithms, and deploy models on SageMaker. When installed, the library defines the following for users:\n+ The locations for storing code and other resources.\n+ The entry point that contains the code to run when the container is started. Your Dockerfile must copy the code that needs to be run into the location expected by a container that is compatible with SageMaker. \n+ Other information that a container needs to manage deployments for training and inference. ",
		}],
	),
	chunk_size: int = Query(default=500, ge=1, examples=[500]),
	chunk_overlap: int = Query(default=50, ge=0, examples=[50]),
	documents: DocumentService = Depends(get_document_service),
) -> SplitTextResponse:
	try:
		chunks = documents.chunk_text(text=payload.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	except ValueError as exc:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

	return SplitTextResponse(count=len(chunks), chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunks=chunks)


@router.get("/local-docs", response_model=LocalDocumentsResponse)
async def list_local_docs(
	documents: DocumentService = Depends(get_document_service),
) -> LocalDocumentsResponse:
	result = documents.list_local_sagemaker_docs()
	return LocalDocumentsResponse.model_validate(result)


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

