"""Pydantic models for DocPilot Web API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class FileStatusEnum(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class StepEnum(StrEnum):
    UPLOADING = "uploading"
    OCR = "ocr"
    DEDUP = "dedup"
    CLASSIFYING = "classifying"
    BUILDING_PDF = "building_pdf"
    DONE = "done"
    ERROR = "error"


class FileStatus(BaseModel):
    name: str
    ext: str
    size: str
    status: FileStatusEnum = FileStatusEnum.QUEUED
    error: str | None = None


class GroupFile(BaseModel):
    name: str
    ext: str
    size: str


class Group(BaseModel):
    name: str
    color: str
    files: list[GroupFile]
    file_count: int
    confidence: int  # 0-100
    method: str  # regex, keyword, gemini


class TaskStatus(BaseModel):
    task_id: str
    step: StepEnum = StepEnum.UPLOADING
    progress: int = 0  # 0-100
    files: list[FileStatus] = []
    message: str = ""


class TaskResult(BaseModel):
    task_id: str
    groups: list[Group] = []
    unclassified_count: int = 0
    total_files: int = 0
    duplicates_removed: int = 0
    processing_time: float = 0.0


class UploadResponse(BaseModel):
    task_id: str
    file_count: int
