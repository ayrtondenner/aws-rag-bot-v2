from __future__ import annotations

import logging
from pathlib import Path

import aiohttp
from fastapi import FastAPI, Request

from app.services.config import S3Config
from app.services.s3_service import S3Service
from app.services.setup.s3_setup_service import S3SetupService

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _docs_dir() -> Path:
    return _project_root() / "sagemaker-docs"


def get_s3_service() -> S3Service:
    """FastAPI dependency provider for an S3Service instance."""

    return S3Service(S3Config.from_env())


def get_http_session_from_app(app: FastAPI) -> aiohttp.ClientSession:
    session = getattr(app.state, "http_session", None)
    if session is None:
        raise RuntimeError("HTTP session not initialized (app.state.http_session)")
    if not isinstance(session, aiohttp.ClientSession):
        raise RuntimeError("Unexpected http_session type")
    return session


def get_http_session(request: Request) -> aiohttp.ClientSession:
    return get_http_session_from_app(request.app)


def get_s3_setup_service() -> S3SetupService:
    return S3SetupService(s3=get_s3_service())