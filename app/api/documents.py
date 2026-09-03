import uuid
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, status
from pydantic import BaseModel

from app.services.ingestion.indexer import DocumentIndexer

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger("kognit.documents")
indexer = DocumentIndexer()

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Ephemeral Job Tracker for MVP (Transition to PostgreSQL in production)
INGESTION_JOBS: Dict[str, Dict[str, Any]] = {}

def process_pdf_background(
    job_id: str,
    stored_path: Path,
    original_filename: str,
    course_code: str,
    unit: int,
    visibility: str
):
    INGESTION_JOBS[job_id]["status"] = "PROCESSING"
    try:
        chunks = indexer.index_pdf(
            pdf_path=stored_path,
            course_code=course_code,
            unit=unit,
            visibility=visibility
        )
        INGESTION_JOBS[job_id]["status"] = "INDEXED"
        INGESTION_JOBS[job_id]["chunks_indexed"] = chunks
        logger.info("Indexed %s successfully (%d chunks)", original_filename, chunks)
    except Exception as e:
        INGESTION_JOBS[job_id]["status"] = "FAILED"
        INGESTION_JOBS[job_id]["error"] = str(e)
        logger.error("Ingestion failed for %s: %s", original_filename, e, exc_info=True)

@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    course_code: str = Form(...),
    unit: int = Form(...),
    visibility: str = Form("global"),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Prevent directory traversal attacks
    safe_basename = Path(file.filename).name
    unique_disk_name = f"{uuid.uuid4().hex[:8]}_{safe_basename}"
    destination_path = UPLOAD_DIR / unique_disk_name

    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    job_id = str(uuid.uuid4())
    INGESTION_JOBS[job_id] = {
        "job_id": job_id,
        "filename": safe_basename,
        "stored_as": unique_disk_name,
        "course_code": course_code.upper(),
        "unit": unit,
        "status": "QUEUED",
        "chunks_indexed": 0
    }

    
    background_tasks.add_task(
        process_pdf_background,
        job_id=job_id,
        stored_path=destination_path,
        original_filename=safe_basename,
        course_code=course_code,
        unit=unit,
        visibility=visibility
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "filename": safe_basename,
        "message": "File received. Processing in background."
    }

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = INGESTION_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return job

@router.get("")
async def list_documents(course_code: Optional[str] = Query(None)):
    try:
        docs = indexer.get_indexed_documents(course_code=course_code)
        return {"status": "success", "total_documents": len(docs), "documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("")
async def delete_document(
    source: str = Query(...), 
    course_code: Optional[str] = Query(None)
):
    result = indexer.delete_by_source(source_filename=source, course_code=course_code)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=result["message"])
    return result