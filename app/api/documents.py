import os
import shutil
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status
from pydantic import BaseModel
from app.services.ingestion.indexer import DocumentIndexer

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger("kognit.documents")
indexer = DocumentIndexer()

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class DeleteDocumentResponse(BaseModel):
    source: str
    course_code: Optional[str] = None
    chunks_deleted: int
    file_deleted_from_disk: bool
    status: str
    message: str


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
            visibility=visibility,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return {
        "status": "success",
        "filename": file.filename,
        "course_code": course_code.upper(),
        "unit": unit,
        "chunks_indexed": chunks_indexed,
    }


@router.delete(
    "",
    response_model=DeleteDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete all indexed vectors and stored files for a document",
)
async def delete_document_endpoint(
    source: str = Query(
        ...,
        description="Exact filename as stored in metadata (e.g., 'Sample_COA.pdf')",
        example="Sample_COA.pdf",
    ),
    course_code: Optional[str] = Query(
        None,
        description="Optional course code filter (e.g., 'COA', 'DBMS')",
        example="COA",
    ),
):
    try:
        # 1. Delete vector points from Qdrant
        result = indexer.delete_by_source(
            source_filename=source, 
            course_code=course_code
        )

        if result["status"] == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["message"],
            )

        # 2. Clean up physical file in storage/uploads if present
        disk_file = UPLOAD_DIR / source
        file_removed = False
        if disk_file.exists() and disk_file.is_file():
            try:
                disk_file.unlink()
                file_removed = True
            except OSError as fs_err:
                logger.warning("Could not delete file %s from disk: %s", disk_file, fs_err)

        logger.info(
            "Purged document '%s' from Qdrant (%d chunks) | Disk file removed: %s",
            source, result["deleted"], file_removed
        )

        return DeleteDocumentResponse(
            source=source,
            course_code=course_code,
            chunks_deleted=result["deleted"],
            file_deleted_from_disk=file_removed,
            status=result["status"],
            message=result["message"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document '%s': %s", source, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during deletion: {str(e)}",
        )
@router.get(
    "",
    summary="List all indexed documents grouped with chunk counts",
)
async def list_documents_endpoint(
    course_code: Optional[str] = Query(
        None,
        description="Optional filter by course code",
        example="DBMS",
    )
):
    try:
        documents = indexer.get_indexed_documents(course_code=course_code)
        return {
            "status": "success",
            "total_documents": len(documents),
            "documents": documents,
        }
    except Exception as e:
        logger.error("Failed to list indexed documents: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query knowledge base: {str(e)}",
        )