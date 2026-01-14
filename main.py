import logging
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status

from app.routes.s3 import router as s3_router
from app.routes.document import router as document_router
from app.services.dependencies import (
    get_s3_setup_service,
)
from app.services.s3_service import S3ServiceError

# TODO: try to implement an agent for chatbot conversations


def _ensure_logging() -> None:
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        root.setLevel(logging.INFO)
        for handler in root.handlers:
            handler.setFormatter(formatter)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_logging()

    # NOTE: FastAPI lifespan runs with only the `app` instance (no per-request `Request`).
    # Some dependency helpers (like `get_opensearch_setup_service(request)`) require a Request
    # because they pull shared resources from `request.app.state`.
    #
    # To keep a single, reusable HTTP client for the whole app lifetime, we create an
    # aiohttp.ClientSession here and store it on `app.state`. App-level dependency helpers
    # can then retrieve it (without needing a Request) during startup.

    # ATTENTION:
    # We create one aiohttp.ClientSession for the app's lifetime and store it on `app.state`.
    # This is *singleton-like* (a single shared instance for reuse), but it is not the strict
    # GoF Singleton pattern:
    # - It's app-scoped (one per FastAPI app instance), not a globally enforced single instance.
    # - In tests or multi-worker deployments, you can still have multiple sessions (one per process).
    #
    # App-level dependency helpers can retrieve the session without needing a Request.
    app.state.http_session = aiohttp.ClientSession()

    # Provision required infrastructure at startup.
    try:
        # Provision infra.
        await get_s3_setup_service().setup_bucket()
        yield
    finally:
        await app.state.http_session.close()


app = FastAPI(lifespan=lifespan)

app.include_router(s3_router)
app.include_router(document_router)


@app.exception_handler(S3ServiceError)
async def s3_service_error_handler(request: Request, exc: S3ServiceError) -> JSONResponse:
    """Map S3 service-layer failures to a consistent HTTP response.

    This keeps AWS/S3 errors from leaking internal details to API consumers while still
    returning a predictable payload the frontend/clients can handle.

    Returns:
        502 Bad Gateway with a JSON body: {"detail": "..."}
    """
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


@app.get("/")
async def root():
    return {"message": "Hello World! AWS RAG Bot is running."}
