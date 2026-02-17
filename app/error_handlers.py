"""Centralised exception-to-HTTP-response handlers for the FastAPI app.

Each handler maps a service-layer error to a consistent JSON response so that
internal details never leak to API consumers.

Usage (in main.py or tests):

    from app.error_handlers import register_error_handlers

    app = FastAPI(...)
    register_error_handlers(app)
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status

from app.services.opensearch_service import OpenSearchServiceError
from app.services.s3_service import S3ServiceError


async def s3_service_error_handler(request: Request, exc: S3ServiceError) -> JSONResponse:  # noqa: ARG001
    """Map S3 service-layer failures to a consistent HTTP response.

    This keeps AWS/S3 errors from leaking internal details to API consumers
    while still returning a predictable payload the frontend/clients can
    handle.

    Returns:
        502 Bad Gateway with a JSON body: ``{"detail": "..."}``
    """
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


async def opensearch_service_error_handler(
    request: Request, exc: OpenSearchServiceError,
) -> JSONResponse:  # noqa: ARG001
    """Map OpenSearch service-layer failures to HTTP 502.

    Returns:
        502 Bad Gateway with a JSON body: ``{"detail": "..."}``
    """
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach all service-error handlers to *app*.

    Args:
        app: The FastAPI application instance.
    """
    app.add_exception_handler(S3ServiceError, s3_service_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(OpenSearchServiceError, opensearch_service_error_handler)  # type: ignore[arg-type]
