import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf
from fastembed import SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ecs_knowledge_base"

DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"

EMBEDDING_VERSION = "v1"
CHUNKING_VERSION = "v1"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 70


def calculate_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def generate_document_id(
    content_hash: str,
    course_code: str,
) -> str:
    raw = f"{course_code.upper()}::{content_hash}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:32]


class DocumentIndexer:

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
    ):
        self.client = QdrantClient(
            url=qdrant_url
        )

        self.collection_name = collection_name

        self.dense_model = SentenceTransformer(
            DENSE_MODEL_NAME
        )

        self.sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        self._ensure_collection()

    def _ensure_collection(self):

        collections = self.client.get_collections()

        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.dense_model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False
                    )
                )
            },
        )

    def extract_text_from_pdf(
        self,
        pdf_path: str,
    ) -> List[Dict[str, Any]]:

        document = None

        try:
            document = pymupdf.open(pdf_path)

            pages = []

            for page_number, page in enumerate(
                document,
                start=1,
            ):

                text = page.get_text(
                    "text"
                )

                if not text:
                    continue

                pages.append(
                    {
                        "page": page_number,
                        "text": text.strip(),
                    }
                )

            return pages

        finally:

            if document is not None:
                document.close()

    def chunk_text(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> List[str]:

        text = text.strip()

        if not text:
            return []

        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(
                start + chunk_size,
                text_length,
            )

            chunk = text[
                start:end
            ].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - overlap

        return chunks

    def document_exists(
        self,
        content_hash: str,
        course_code: Optional[str] = None,
    ) -> bool:

        must_conditions = [
            FieldCondition(
                key="content_hash",
                match=MatchValue(
                    value=content_hash
                ),
            )
        ]

        if course_code:

            must_conditions.append(
                FieldCondition(
                    key="course_code",
                    match=MatchValue(
                        value=course_code.upper()
                    ),
                )
            )

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=must_conditions
            ),
            exact=True,
        )

        return result.count > 0

    def index_pdf(
        self,
        pdf_path: str,
        course_code: str,
        unit: int,
        visibility: str = "global",
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:

        pdf_path = str(pdf_path)

        stored_filename = Path(
            pdf_path
        ).name

        source_filename = (
            original_filename
            or stored_filename
        )

        normalized_course = course_code.upper()

        content_hash = calculate_file_hash(
            pdf_path
        )

        document_id = generate_document_id(
            content_hash,
            normalized_course,
        )

        if self.document_exists(
            content_hash=content_hash,
            course_code=normalized_course,
        ):

            return {
                "status": "duplicate",
                "document_id": document_id,
                "content_hash": content_hash,
                "chunks_indexed": 0,
                "source": source_filename,
            }

        pages = self.extract_text_from_pdf(
            pdf_path
        )

        points = []

        chunk_index = 0

        for page_data in pages:

            page_number = page_data["page"]
            page_text = page_data["text"]

            chunks = self.chunk_text(
                page_text
            )

            for chunk in chunks:

                dense_vector = self.dense_model.encode(
                    chunk,
                    normalize_embeddings=True,
                ).tolist()

                sparse_embedding = next(
                    self.sparse_model.embed(
                        [chunk]
                    )
                )

                sparse_vector = SparseVector(
                    indices=sparse_embedding.indices.tolist(),
                    values=sparse_embedding.values.tolist(),
                )

                payload = {
                    "text": chunk,
                    "document_id": document_id,
                    "content_hash": content_hash,
                    "source": source_filename,
                    "original_filename": source_filename,
                    "stored_filename": stored_filename,
                    "course_code": normalized_course,
                    "unit": unit,
                    "visibility": visibility,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "source_type": "course",
                    "embedding_model": DENSE_MODEL_NAME,
                    "embedding_version": EMBEDDING_VERSION,
                    "chunking_version": CHUNKING_VERSION,
                }

                point = PointStruct(
                    id=hash(
                        f"{document_id}:{chunk_index}"
                    ) & 0x7FFFFFFFFFFFFFFF,
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_vector,
                    },
                    payload=payload,
                )

                points.append(point)

                chunk_index += 1

        if points:

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        logger.info(
            "Indexed %s: %s chunks",
            source_filename,
            len(points),
        )

        return {
            "status": "success",
            "document_id": document_id,
            "content_hash": content_hash,
            "source": source_filename,
            "stored_filename": stored_filename,
            "course_code": normalized_course,
            "unit": unit,
            "chunks_indexed": len(points),
        }

    def get_indexed_documents(
        self,
    ) -> List[Dict[str, Any]]:

        records = {}

        offset = None

        while True:

            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:

                payload = point.payload or {}

                document_id = payload.get(
                    "document_id"
                )

                source = payload.get(
                    "source",
                    "Unknown",
                )

                course_code = payload.get(
                    "course_code",
                    "UNKNOWN",
                )

                key = (
                    f"{course_code}::"
                    f"{source}::"
                    f"{document_id}"
                )

                if key not in records:

                    records[key] = {
                        "document_id": document_id,
                        "content_hash": payload.get(
                            "content_hash"
                        ),
                        "source": source,
                        "original_filename": payload.get(
                            "original_filename",
                            source,
                        ),
                        "stored_filename": payload.get(
                            "stored_filename"
                        ),
                        "course_code": course_code,
                        "unit": payload.get(
                            "unit"
                        ),
                        "embedding_model": payload.get(
                            "embedding_model"
                        ),
                        "embedding_version": payload.get(
                            "embedding_version"
                        ),
                        "chunking_version": payload.get(
                            "chunking_version"
                        ),
                        "chunk_count": 0,
                    }

                records[key]["chunk_count"] += 1

            if offset is None:
                break

        return list(
            records.values()
        )

    def delete_by_source(
        self,
        source_filename: str,
        course_code: Optional[str] = None,
    ) -> Dict[str, Any]:

        must_conditions = [
            FieldCondition(
                key="source",
                match=MatchValue(
                    value=source_filename
                ),
            )
        ]

        if course_code:

            must_conditions.append(
                FieldCondition(
                    key="course_code",
                    match=MatchValue(
                        value=course_code.upper()
                    ),
                )
            )

        target_filter = Filter(
            must=must_conditions
        )

        points_count = self.client.count(
            collection_name=self.collection_name,
            count_filter=target_filter,
            exact=True,
        ).count

        if points_count == 0:

            return {
                "deleted": 0,
                "status": "not_found",
                "message": "No matching document found.",
            }

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=target_filter
            ),
        )

        return {
            "deleted": points_count,
            "status": "success",
            "message": "Document deleted successfully.",
        }


indexer = DocumentIndexer()