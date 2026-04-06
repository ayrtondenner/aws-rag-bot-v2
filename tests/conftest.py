import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.error_handlers import register_error_handlers
from app.routes.document import router as document_router
from app.routes.search import router as search_router
from app.routes.s3 import router as s3_router

@pytest.fixture(scope='session')
def sample_fixture():
    return "Hello, World!"


@pytest.fixture()
def fastapi_app() -> FastAPI:
    """A minimal FastAPI app for route tests.

    Notes:
    - Does NOT use the production lifespan (avoids S3 provisioning during tests).
    - Includes the same exception handlers as main.py via register_error_handlers().
    """

    app = FastAPI()
    app.include_router(s3_router)
    app.include_router(document_router)
    app.include_router(search_router)
    register_error_handlers(app)

    return app


@pytest.fixture()
def client(fastapi_app: FastAPI):
    with TestClient(fastapi_app) as c:
        yield c
