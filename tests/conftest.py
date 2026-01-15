import pytest

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette import status

from app.routes.document import router as document_router
from app.routes.s3 import router as s3_router
from app.services.s3_service import S3ServiceError

@pytest.fixture(scope='session')
def sample_fixture():
    return "Hello, World!"


@pytest.fixture()
def fastapi_app() -> FastAPI:
    """A minimal FastAPI app for route tests.

    Notes:
    - Does NOT use the production lifespan (avoids S3 provisioning during tests).
    - Includes the same S3ServiceError exception handler as main.py.
    """

    app = FastAPI()
    app.include_router(s3_router)
    app.include_router(document_router)

    @app.exception_handler(S3ServiceError)
    async def s3_service_error_handler(request: Request, exc: S3ServiceError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})

    return app


@pytest.fixture()
def client(fastapi_app: FastAPI):
    with TestClient(fastapi_app) as c:
        yield c