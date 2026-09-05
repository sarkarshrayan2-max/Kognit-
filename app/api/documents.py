import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, get_db
from app.models.course import Course
from app.models.document import Document
from app.services.ingestion.indexer import (
    CHUNKING_VERSION,
    COLLECTION_NAME,
    DENSE_MODEL_NAME,
    EMBEDDING_VERSION,
    generate_document_id,
    indexer,
)

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

INGESTION_JOBS: dict[str, dict[str, Any]] = {}


def safe_filename(filename: str) -> str:
    """
    Prevent path traversal and normalize unsafe filenames.
    """
    filename = Path(filename).name

    filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        filename,
    )

    if not filename:
        filename = "document.pdf"

    return filename


def resolve_stored_pdf(
    stored_filename: str,
) -> Path:
    """
    Resolve a stored PDF safely inside uploads/.
    """
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


def get_course(
    db: Session,
    course_code: str,
) -> Course:
    """
    Retrieve a course from PostgreSQL.
    """
    course = db.scalar(
        select(Course).where(
            Course.code == course_code
        )
    )

    if course is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Course '{course_code}' not found."
            ),
        )

    return course


def process_pdf_background(
    job_id: str,
    pdf_path: str,
    original_filename: str,
    course_code: str,
    unit: int,
    visibility: str,
):
    db = SessionLocal()

    try:
        INGESTION_JOBS[job_id] = {
            "status": "processing",
            "filename": original_filename,
            "stored_filename": Path(
                pdf_path
            ).name,
            "course_code": course_code,
            "unit": unit,
        }

        pdf = Path(pdf_path)

        if not pdf.exists():
            raise FileNotFoundError(
                f"PDF file does not exist: {pdf_path}"
            )

        result = indexer.index_pdf(
            pdf_path=str(pdf),
            course_code=course_code,
            unit=unit,
            visibility=visibility,
            original_filename=original_filename,
        )

        if result["status"] == "duplicate":
            INGESTION_JOBS[job_id] = {
                "status": "duplicate",
                **result,
                "filename": original_filename,
                "stored_filename": pdf.name,
            }

            try:
                pdf.unlink()
            except FileNotFoundError:
                pass

            return

        document_id = result["document_id"]
        content_hash = result["content_hash"]
        chunk_count = result.get(
            "chunks_indexed",
            0,
        )
        stored_filename = result.get(
            "stored_filename",
            pdf.name,
        )

        course = get_course(
            db,
            course_code,
        )

        document = db.scalar(
            select(Document).where(
                Document.document_id == document_id
            )
        )

        if document is None:
            document = Document(
                document_id=document_id,
                course_id=course.id,
                original_filename=original_filename,
                stored_filename=stored_filename,
                content_hash=content_hash,
                unit=unit,
                visibility=visibility,
                chunk_count=chunk_count,
                embedding_model=DENSE_MODEL_NAME,
                embedding_version=EMBEDDING_VERSION,
                chunking_version=CHUNKING_VERSION,
                qdrant_collection=COLLECTION_NAME,
            )
            db.add(document)
        else:
            document.original_filename = original_filename
            document.stored_filename = stored_filename
            document.chunk_count = chunk_count
            document.unit = unit
            document.visibility = visibility

        db.commit()

        INGESTION_JOBS[job_id] = {
            "status": "completed",
            **result,
        }

        logger.info(
            "Successfully indexed PDF '%s' with %s chunks.",
            original_filename,
            chunk_count,
        )

    except Exception as exc:
        db.rollback()

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
            "course_code": course_code,
            "unit": unit,
            "error": str(exc),
        }

    finally:
        db.close()


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    course_code: str = Form(...),
    unit: int = Form(...),
    visibility: str = Form("global"),
    db: Session = Depends(get_db),
):
    normalized_course = (
        course_code.strip().upper()
    )

    if not normalized_course:
        raise HTTPException(
            status_code=400,
            detail="Course code cannot be empty.",
        )

    course = get_course(
        db,
        normalized_course,
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

    existing_document = db.scalar(
        select(Document).where(
            Document.course_id == course.id,
            Document.content_hash == content_hash,
        )
    )

    if existing_document is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This document is already registered "
                "for this course."
            ),
        )

    if indexer.document_exists(
        content_hash=content_hash,
        course_code=normalized_course,
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "This document already exists in "
                "the vector database for this course."
            ),
        )

    document_id = generate_document_id(
        content_hash,
        normalized_course,
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
            detail=(
                f"Failed to save PDF: {exc}"
            ),
        )

    if not pdf_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "PDF was uploaded but could not "
                "be stored."
            ),
        )

    INGESTION_JOBS[job_id] = {
        "status": "queued",
        "filename": original_filename,
        "stored_filename": stored_filename,
        "course_code": normalized_course,
        "course_id": course.id,
        "unit": unit,
        "document_id": document_id,
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
        "document_id": document_id,
        "filename": original_filename,
        "stored_filename": stored_filename,
        "course_code": normalized_course,
        "course_id": course.id,
        "unit": unit,
    }


@router.get("/jobs/{job_id}")
def get_ingestion_job(
    job_id: str,
):
    job = INGESTION_JOBS.get(
        job_id
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Ingestion job not found.",
        )

    return job


@router.get("")
def list_documents(
    db: Session = Depends(get_db),
):
    documents = db.scalars(
        select(Document)
        .order_by(
            Document.created_at.desc()
        )
    ).all()

    result = []

    for document in documents:
        result.append(
            {
                "id": document.id,
                "document_id": document.document_id,
                "course_id": document.course_id,
                "course_code": (
                    document.course.code
                    if document.course
                    else None
                ),
                "original_filename": (
                    document.original_filename
                ),
                "stored_filename": (
                    document.stored_filename
                ),
                "content_hash": (
                    document.content_hash
                ),
                "unit": document.unit,
                "visibility": document.visibility,
                "chunk_count": (
                    document.chunk_count
                ),
                "embedding_model": (
                    document.embedding_model
                ),
                "embedding_version": (
                    document.embedding_version
                ),
                "chunking_version": (
                    document.chunking_version
                ),
                "qdrant_collection": (
                    document.qdrant_collection
                ),
                "created_at": (
                    document.created_at
                ),
                "updated_at": (
                    document.updated_at
                ),
            }
        )

    return {
        "documents": result,
        "count": len(result),
    }


@router.get("/files/{document_id}")
def serve_document(
    document_id: str,
    db: Session = Depends(get_db),
):
    document = db.scalar(
        select(Document).where(
            Document.document_id == document_id
        )
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if not document.stored_filename:
        raise HTTPException(
            status_code=404,
            detail="Stored PDF information is unavailable.",
        )

    pdf_path = resolve_stored_pdf(
        document.stored_filename
    )

    if not pdf_path.exists():
        logger.error(
            "PDF missing for document_id=%s",
            document_id,
        )
        raise HTTPException(
            status_code=404,
            detail=(
                "PDF file is no longer available "
                "on the server."
            ),
        )

    if not pdf_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Stored PDF path is not a file.",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=safe_filename(
            document.original_filename
        ),
        headers={
            "Content-Disposition": (
                "inline; "
                f'filename="{safe_filename(document.original_filename)}"'
            )
        },
    )


@router.delete("")
def delete_document(
    source: str,
    course_code: str,
    db: Session = Depends(get_db),
):
    normalized_course = (
        course_code.strip().upper()
    )

    course = get_course(
        db,
        normalized_course,
    )

    document = db.scalar(
        select(Document).where(
            Document.course_id == course.id,
            Document.original_filename == source,
        )
    )

    if document is None:
        documents = (
            indexer.get_indexed_documents()
        )

        matching_document = None

        for item in documents:
            item_source = item.get(
                "source"
            )

            item_course = str(
                item.get(
                    "course_code",
                    "",
                )
            ).upper()

            if (
                item_source == source
                and item_course == normalized_course
            ):
                matching_document = item
                break

        if matching_document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        document_id = matching_document.get(
            "document_id"
        )

        stored_filename = matching_document.get(
            "stored_filename"
        )
    else:
        document_id = document.document_id
        stored_filename = document.stored_filename

    try:
        qdrant_result = indexer.delete_by_source(
            source_filename=source,
            course_code=normalized_course,
        )
    except Exception as exc:
        logger.exception(
            "Failed to delete document from Qdrant."
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to delete document "
                f"from vector database: {exc}"
            ),
        )

    deleted_file = None

    if stored_filename:
        pdf_path = resolve_stored_pdf(
            str(stored_filename)
        )

        try:
            if pdf_path.exists():
                pdf_path.unlink()
                deleted_file = pdf_path.name
        except OSError as exc:
            logger.exception(
                "Failed to delete stored PDF."
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "Vectors were deleted, but "
                    f"the PDF could not be removed: {exc}"
                ),
            )

    if document is not None:
        db.delete(document)
        db.commit()

    return {
        "status": "deleted",
        "message": "Document deleted successfully.",
        "document_id": document_id,
        "source": source,
        "course_code": normalized_course,
        "deleted_file": deleted_file,
        "qdrant_result": qdrant_result,
    }


@router.patch("/{document_id}/course")
def update_document_course(
    document_id: str,
    old_course_code: str,
    new_course_code: str,
):
    result = indexer.update_course_code(
        document_id=document_id,
        old_course_code=old_course_code,
        new_course_code=new_course_code,
    )

    if result["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail=result["message"],
        )

    return result