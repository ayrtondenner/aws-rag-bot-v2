from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class FileItem(BaseModel):
    key: str = Field(..., description="S3 object key")
    size: Optional[int] = None
    last_modified: Optional[datetime] = None
    etag: Optional[str] = None

    @staticmethod
    def from_s3_object(obj: dict[str, Any]) -> "FileItem":
        return FileItem(
            key=str(obj.get("Key")),
            size=obj.get("Size"),
            last_modified=obj.get("LastModified"),
            etag=obj.get("ETag"),
        )

class FileListResponse(BaseModel):
    count: int
    files: list[FileItem]


class BucketExistsResponse(BaseModel):
    bucket_name: str
    exists: bool


class BucketFileCountResponse(BaseModel):
    bucket_name: str
    prefix: str | None = None
    count: int


class S3FileContentResponse(BaseModel):
    filename: str
    content: str
