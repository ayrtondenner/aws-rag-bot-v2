from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.models.s3 import (
    BucketExistsResponse,
    BucketFileCountResponse,
    FileListResponse,
    S3FileContentResponse,
)
from app.services.dependencies import get_s3_service
from app.services.s3_service import S3Service
from fastapi import HTTPException

from starlette import status

router = APIRouter(prefix="/s3", tags=["s3"])


@router.get("/bucket/exists", response_model=BucketExistsResponse,
        responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "bucket_name": "senior-sagemaker-assessment-bucket",
                        "exists": True,
                    }
                }
            }
        }
    },
)
async def bucket_exists(
    s3: S3Service = Depends(get_s3_service),
) -> BucketExistsResponse:
    """Check whether the configured S3 bucket exists (and is accessible)."""

    bucket_name = (os.getenv("S3_BUCKET_NAME") or "").strip()
    if not bucket_name:
        raise ValueError("Missing required environment variable: S3_BUCKET_NAME")

    exists = await s3.bucket_exists(bucket_name=bucket_name)
    return BucketExistsResponse(bucket_name=bucket_name, exists=exists)


@router.get("/bucket/files/count", response_model=BucketFileCountResponse,
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "bucket_name": "senior-sagemaker-assessment-bucket",
                        "prefix": None,
                        "count": 336,
                    }
                }
            }
        }
    },
)
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

@router.get("/files", response_model=FileListResponse,
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "count": 3,
                        "files": [
                            {
                                "key": "sagemaker-docs/amazon-sagemaker-toolkits.md",
                                "size": 4158,
                                "last_modified": "2025-12-26T18:39:51Z",
                                "etag": "\"3b88730829ecf9cb25b9aaea09df297f\""
                            },
                            {
                                "key": "sagemaker-docs/asff-resourcedetails-awssagemaker.md",
                                "size": 1880,
                                "last_modified": "2025-12-26T18:25:23Z",
                                "etag": "\"6cb9adb9a318c3d40028af9fb173fa45\""
                            },
                            {
                                "key": "sagemaker-docs/automating-sagemaker-with-eventbridge.md",
                                "size": 26997,
                                "last_modified": "2025-12-26T18:25:27Z",
                                "etag": "\"533f5ccff4668514e0a33c6cb22d150c\""
                            },
                        ]
                    }
                }
            }
        }
    },
)
async def list_files(
    prefix: Optional[str] = Query(default=None),
    s3: S3Service = Depends(get_s3_service),
) -> FileListResponse:
    """List objects in the configured S3 bucket (optionally filtered by prefix)."""

    files = await s3.list_files(prefix=prefix)
    return FileListResponse(count=len(files), files=files)

@router.get(
    "/file/content",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "filename": "sagemaker-docs/sagemaker-algo-common-data-formats.md",
                        "content": "# Common Data Formats for Built-in Algorithms<a name=\"sagemaker-algo-common-data-formats\"></a>\n\nThe following topics explain the data formats for the algorithms provided by Amazon SageMaker.\n\n**Topics**\n+ [Common Data Formats for Training](cdf-training.md)\n+ [Common Data Formats for Inference](cdf-inference.md)\n",
                    }
                },
            }
        },
        404: {"description": "File not found"},
        500: {"description": "Failed to fetch file content"},
    },
)
async def get_file_content(
    file_name: str = Query(
        ..., min_length=1, examples=["sagemaker-docs/sagemaker-algo-common-data-formats.md"]
    ),
    s3: S3Service = Depends(get_s3_service),
)-> S3FileContentResponse:
    """Get the raw content of an object in the configured S3 bucket by file name (key)."""

    try:
        content = await s3.get_file_content(key=file_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch file content") from exc

    return S3FileContentResponse(filename=file_name, content=content)