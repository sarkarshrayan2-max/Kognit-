import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.ingestion.indexer import DocumentIndexer

router = APIRouter(prefix="/documents", tags=["Documents"])
indexer = DocumentIndexer()

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    course_code: str = Form(...),
    unit: int = Form(...),
    visibility: str = Form("global"),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    
    try:
        chunks_indexed = indexer.index_pdf(
            pdf_path=file_path,
            course_code=course_code,
            unit=unit,
            visibility=visibility
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return {
        "status": "success",
        "filename": file.filename,
        "course_code": course_code.upper(),
        "unit": unit,
        "chunks_indexed": chunks_indexed
    }