from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.models.s3 import (
    BucketExistsResponse,
    BucketFileCountResponse,
    FileListResponse,
)
from app.services.dependencies import get_s3_service
from app.services.s3_service import S3Service
from fastapi import HTTPException
from fastapi.responses import Response

from starlette import status

router = APIRouter(prefix="/s3", tags=["s3"])


@router.get("/bucket/exists", response_model=BucketExistsResponse)
async def bucket_exists(
    s3: S3Service = Depends(get_s3_service),
) -> BucketExistsResponse:
    """Check whether the configured S3 bucket exists (and is accessible)."""

    bucket_name = (os.getenv("S3_BUCKET_NAME") or "").strip()
    if not bucket_name:
        raise ValueError("Missing required environment variable: S3_BUCKET_NAME")

    exists = await s3.bucket_exists(bucket_name=bucket_name)
    return BucketExistsResponse(bucket_name=bucket_name, exists=exists)


@router.get("/bucket/files/count", response_model=BucketFileCountResponse)
async def bucket_files_count(
    prefix: Optional[str] = Query(default=None),
    s3: S3Service = Depends(get_s3_service),
) -> BucketFileCountResponse:
    """Return a count of objects in the configured S3 bucket (optionally filtered by prefix)."""

    bucket_name = (os.getenv("S3_BUCKET_NAME") or "").strip()
    if not bucket_name:
        raise ValueError("Missing required environment variable: S3_BUCKET_NAME")

    files = await s3.list_files(prefix=prefix)
    return BucketFileCountResponse(bucket_name=bucket_name, prefix=prefix, count=len(files))


@router.get("/files", response_model=FileListResponse)
async def list_files(
    prefix: Optional[str] = Query(default=None),
    s3: S3Service = Depends(get_s3_service),
) -> FileListResponse:
    """List objects in the configured S3 bucket (optionally filtered by prefix)."""

    files = await s3.list_files(prefix=prefix)
    return FileListResponse(count=len(files), files=files)

@router.get("/file/content")
async def get_file_content(
    file_name: str = Query(..., min_length=1),
    s3: S3Service = Depends(get_s3_service),
):
    """Get the raw content of an object in the configured S3 bucket by file name (key)."""

    try:
        content = await s3.get_file_content(key=file_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch file content") from exc

    if isinstance(content, str):
        return Response(content=content, media_type="text/plain; charset=utf-8")

    return Response(content=content, media_type="application/octet-stream")