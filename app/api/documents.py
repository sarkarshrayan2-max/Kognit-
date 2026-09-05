import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.services.ingestion.indexer import DocumentIndexer

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

logger = logging.getLogger("kognit.documents")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_UPLOAD_SIZE = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
}

INGESTION_JOBS: Dict[str, Dict[str, Any]] = {}

indexer = DocumentIndexer()


def safe_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        filename,
    )

    if not filename:
        filename = "document.pdf"

    return filename


def resolve_stored_pdf(stored_filename: str) -> Path:
    filename = Path(stored_filename).name

    pdf_path = (
        UPLOAD_DIR / filename
    ).resolve()

    upload_root = UPLOAD_DIR.resolve()

    if pdf_path.parent != upload_root:
        raise HTTPException(
            status_code=403,
            detail="Invalid document path.",
        )

    return pdf_path


def find_document_point(document_id: str):
    points, _ = indexer.client.scroll(
        collection_name=indexer.collection_name,
        scroll_filter={
            "must": [
                {
                    "key": "document_id",
                    "match": {
                        "value": document_id,
                    },
                }
            ]
        },
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return None

    return points[0]


def process_pdf_background(
    job_id: str,
    pdf_path: str,
    original_filename: str,
    course_code: str,
    unit: int,
    visibility: str,
):
    try:
        INGESTION_JOBS[job_id] = {
            "status": "processing",
            "filename": original_filename,
            "stored_filename": Path(
                pdf_path
            ).name,
            "course_code": course_code.upper(),
            "unit": unit,
        }

        if not Path(pdf_path).exists():
            raise FileNotFoundError(
                f"PDF file does not exist: {pdf_path}"
            )

        result = indexer.index_pdf(
            pdf_path=pdf_path,
            course_code=course_code,
            unit=unit,
            visibility=visibility,
            original_filename=original_filename,
        )

        if result["status"] == "duplicate":
            INGESTION_JOBS[job_id] = {
                "status": "duplicate",
                **result,
                "stored_filename": Path(
                    pdf_path
                ).name,
            }
            return

        INGESTION_JOBS[job_id] = {
            "status": "completed",
            **result,
            "stored_filename": Path(
                pdf_path
            ).name,
        }

        logger.info(
            "Successfully indexed PDF: %s",
            original_filename,
        )

    except Exception as exc:
        logger.exception(
            "PDF ingestion failed: %s",
            original_filename,
        )

        INGESTION_JOBS[job_id] = {
            "status": "failed",
            "filename": original_filename,
            "stored_filename": Path(
                pdf_path
            ).name,
            "course_code": course_code.upper(),
            "unit": unit,
            "error": str(exc),
        }


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    course_code: str = Form(...),
    unit: int = Form(...),
    visibility: str = Form("global"),
):
    normalized_course = (
        course_code.strip().upper()
    )

    if not normalized_course:
        raise HTTPException(
            status_code=400,
            detail="Course code cannot be empty.",
        )

    original_filename = safe_filename(
        file.filename or "document.pdf"
    )

    extension = Path(
        original_filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty.",
        )

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum PDF size is 25 MB.",
        )

    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid PDF.",
        )

    content_hash = hashlib.sha256(
        content
    ).hexdigest()

    if indexer.document_exists(
        content_hash=content_hash,
        course_code=normalized_course,
    ):
        raise HTTPException(
            status_code=409,
            detail="This document is already indexed for this course.",
        )

    job_id = hashlib.sha256(
        f"{content_hash}:{normalized_course}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    stored_filename = (
        f"{job_id}_{original_filename}"
    )

    pdf_path = (
        UPLOAD_DIR / stored_filename
    )

    try:
        with open(
            pdf_path,
            "wb",
        ) as f:
            f.write(content)

    except OSError as exc:
        logger.exception(
            "Failed to save uploaded PDF."
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save PDF: {exc}",
        )

    if not pdf_path.exists():
        raise HTTPException(
            status_code=500,
            detail="PDF was uploaded but could not be stored.",
        )

    INGESTION_JOBS[job_id] = {
        "status": "queued",
        "filename": original_filename,
        "stored_filename": stored_filename,
        "course_code": normalized_course,
        "unit": unit,
    }

    background_tasks.add_task(
        process_pdf_background,
        job_id,
        str(pdf_path),
        original_filename,
        normalized_course,
        unit,
        visibility,
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "filename": original_filename,
        "stored_filename": stored_filename,
        "course_code": normalized_course,
        "unit": unit,
    }


@router.get("/jobs/{job_id}")
def get_ingestion_job(
    job_id: str,
):
    job = INGESTION_JOBS.get(
        job_id
    )

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Ingestion job not found.",
        )

    return job


@router.get("")
def list_documents():
    documents = (
        indexer.get_indexed_documents()
    )

    return {
        "documents": documents,
        "count": len(documents),
    }


@router.get("/files/{document_id}")
def serve_document(
    document_id: str,
):
    point = find_document_point(
        document_id
    )

    if point is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    payload = (
        point.payload or {}
    )

    stored_filename = payload.get(
        "stored_filename"
    )

    if not stored_filename:
        raise HTTPException(
            status_code=404,
            detail="Stored PDF information is unavailable.",
        )

    original_filename = payload.get(
        "original_filename"
    )

    if not original_filename:
        original_filename = payload.get(
            "source",
            "document.pdf",
        )

    original_filename = safe_filename(
        str(original_filename)
    )

    pdf_path = resolve_stored_pdf(
        str(stored_filename)
    )

    if not pdf_path.exists():
        logger.error(
            "Qdrant document exists but PDF is missing. "
            "document_id=%s stored_filename=%s expected_path=%s",
            document_id,
            stored_filename,
            pdf_path,
        )

        raise HTTPException(
            status_code=404,
            detail="PDF file is no longer available on the server.",
        )

    if not pdf_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Stored PDF path is not a file.",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=original_filename,
        headers={
            "Content-Disposition": (
                f'inline; filename="{original_filename}"'
            )
        },
    )


@router.delete("")
def delete_document(
    source: str,
    course_code: str,
):
    normalized_course = (
        course_code.strip().upper()
    )

    documents_to_remove = []

    try:
        documents = (
            indexer.get_indexed_documents()
        )

        for document in documents:
            document_source = document.get(
                "source"
            )

            document_course = str(
                document.get(
                    "course_code",
                    "",
                )
            ).upper()

            if (
                document_source == source
                and document_course == normalized_course
            ):
                documents_to_remove.append(
                    document
                )

    except Exception as exc:
        logger.exception(
            "Failed to find document before deletion."
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to locate document: {exc}",
        )

    result = indexer.delete_by_source(
        source_filename=source,
        course_code=normalized_course,
    )

    if result["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail=result["message"],
        )

    deleted_files = []

    for document in documents_to_remove:
        stored_filename = document.get(
            "stored_filename"
        )

        if not stored_filename:
            continue

        try:
            pdf_path = resolve_stored_pdf(
                str(stored_filename)
            )

            if pdf_path.exists():
                pdf_path.unlink()
                deleted_files.append(
                    pdf_path.name
                )

        except Exception:
            logger.exception(
                "Failed to delete stored PDF: %s",
                stored_filename,
            )

    return {
        **result,
        "source": source,
        "course_code": normalized_course,
        "deleted_files": deleted_files,
    }