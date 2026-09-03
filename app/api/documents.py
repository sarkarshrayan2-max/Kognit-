import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.services.ingestion.indexer import (
    DocumentIndexer,
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

logger = logging.getLogger("kognit.documents")

UPLOAD_DIR = Path("uploads")
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




def calculate_sha256(file_path: Path) -> str:

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def safe_filename(filename: str) -> str:

    filename = Path(filename).name

    filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        filename,
    )

    return filename




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
            "course_code": course_code,
        }

        result = indexer.index_pdf(
            pdf_path=pdf_path,
            course_code=course_code,
            unit=unit,
            visibility=visibility,
            original_filename=original_filename,
        )

        if result["status"] == "duplicate":

            try:
                os.remove(pdf_path)
            except OSError:
                pass

            INGESTION_JOBS[job_id] = {
                "status": "duplicate",
                **result,
            }

            return

        INGESTION_JOBS[job_id] = {
            "status": "completed",
            **result,
        }

    except Exception as exc:

        logger.exception(
            "PDF ingestion failed"
        )

        try:
            os.remove(pdf_path)
        except OSError:
            pass

        INGESTION_JOBS[job_id] = {
            "status": "failed",
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
        course_code=course_code,
    ):

        raise HTTPException(
            status_code=409,
            detail="This document is already indexed for this course.",
        )

    

    job_id = hashlib.sha256(
        f"{content_hash}:{course_code}".encode()
    ).hexdigest()[:16]

    stored_filename = (
        f"{job_id}_{original_filename}"
    )

    pdf_path = (
        UPLOAD_DIR / stored_filename
    )

    

    with open(pdf_path, "wb") as f:
        f.write(content)

    INGESTION_JOBS[job_id] = {
        "status": "queued",
        "filename": original_filename,
        "stored_filename": stored_filename,
        "course_code": course_code.upper(),
        "unit": unit,
    }

    

    background_tasks.add_task(
        process_pdf_background,
        job_id,
        str(pdf_path),
        original_filename,
        course_code,
        unit,
        visibility,
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "filename": original_filename,
        "course_code": course_code.upper(),
    }




@router.get("/jobs/{job_id}")
def get_ingestion_job(
    job_id: str,
):

    job = INGESTION_JOBS.get(job_id)

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

    points, _ = indexer.client.scroll(
        collection_name=indexer.collection_name,
        scroll_filter={
            "must": [
                {
                    "key": "document_id",
                    "match": {
                        "value": document_id
                    },
                }
            ]
        },
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    payload = points[0].payload or {}

    stored_filename = payload.get(
        "stored_filename"
    )

    original_filename = payload.get(
        "original_filename",
        payload.get(
            "source",
            "document.pdf",
        ),
    )

    if not stored_filename:

        raise HTTPException(
            status_code=404,
            detail="Stored PDF information is unavailable.",
        )

    stored_filename = Path(
        stored_filename
    ).name

    pdf_path = (
        UPLOAD_DIR / stored_filename
    ).resolve()

    upload_root = (
        UPLOAD_DIR
        .resolve()
    )

    
    if upload_root not in pdf_path.parents:

        raise HTTPException(
            status_code=403,
            detail="Invalid document path.",
        )

    if not pdf_path.exists():

        raise HTTPException(
            status_code=404,
            detail="PDF file is no longer available on the server.",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=original_filename,
        headers={
            "Content-Disposition": (
                f'inline; filename="{safe_filename(original_filename)}"'
            )
        },
    )




@router.delete("")
def delete_document(
    source: str,
    course_code: str,
):

    result = indexer.delete_by_source(
        source_filename=source,
        course_code=course_code,
    )

    if result["status"] == "not_found":

        raise HTTPException(
            status_code=404,
            detail=result["message"],
        )

    return result