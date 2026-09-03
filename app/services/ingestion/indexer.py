import uuid
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

COLLECTION_NAME = "ecs_knowledge_base"
DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"


class DocumentIndexer:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.collection_name = COLLECTION_NAME
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=70,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=1024,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )

    @staticmethod
    def extract_text_from_pdf(pdf_path: str | Path) -> List[Dict[str, Any]]:
        doc = pymupdf.open(str(pdf_path))
        pages_content = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages_content.append({"page": page_num + 1, "text": text})
        return pages_content

    def index_pdf(
        self,
        pdf_path: str | Path,
        course_code: str,
        unit: int,
        visibility: str = "global",
    ) -> int:
        pages = self.extract_text_from_pdf(pdf_path)
        all_chunks: List[Dict[str, Any]] = []

        for p in pages:
            chunks = self.text_splitter.split_text(p["text"])
            for idx, chunk in enumerate(chunks):
                if len(chunk.strip()) > 30:
                    all_chunks.append(
                        {
                            "text": chunk,
                            "metadata": {
                                "source": Path(pdf_path).name,
                                "course_code": course_code.upper(),
                                "unit": unit,
                                "page": p["page"],
                                "chunk_index": idx,
                                "visibility": visibility,
                            },
                        }
                    )

        if not all_chunks:
            return 0

        raw_texts = [c["text"] for c in all_chunks]

        dense_embeddings = self.dense_model.encode(
            raw_texts, normalize_embeddings=True
        ).tolist()
        sparse_embeddings = list(self.sparse_model.embed(raw_texts))

        points = []
        for i, item in enumerate(all_chunks):
            sparse_val = sparse_embeddings[i]
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_embeddings[i],
                        "sparse": SparseVector(
                            indices=sparse_val.indices.tolist(),
                            values=sparse_val.values.tolist(),
                        ),
                    },
                    payload={"text": item["text"], **item["metadata"]},
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def delete_by_source(
        self, source_filename: str, course_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deletes all vector points matching 'source' (and optionally 'course_code').
        Matches the flattened payload keys produced by index_pdf().
        """
        must_conditions = [
            FieldCondition(
                key="source",
                match=MatchValue(value=source_filename),
            )
        ]

        if course_code:
            must_conditions.append(
                FieldCondition(
                    key="course_code",
                    match=MatchValue(value=course_code.upper()),
                )
            )

        target_filter = Filter(must=must_conditions)

        # Count existing chunks to report back
        points_count = self.client.count(
            collection_name=self.collection_name,
            count_filter=target_filter,
            exact=True,
        ).count

        if points_count == 0:
            return {
                "deleted": 0,
                "status": "not_found",
                "message": f"No points found for document '{source_filename}'"
                + (f" under course '{course_code.upper()}'" if course_code else ""),
            }

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=target_filter),
        )

        return {
            "deleted": points_count,
            "status": "success",
            "message": f"Successfully deleted {points_count} chunks for document '{source_filename}'",
        }
    def get_indexed_documents(self, course_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scrolls through the Qdrant collection, groups chunks by document source,
        and aggregates chunk counts, course codes, and unit numbers.
        """
        scroll_filter = None
        if course_code:
            scroll_filter = Filter(
                must=[FieldCondition(key="course_code", match=MatchValue(value=course_code.upper()))]
            )

        docs_summary: Dict[str, Dict[str, Any]] = {}
        offset = None

        while True:
            scroll_result, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=250,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )

            for point in scroll_result:
                payload = point.payload or {}
                source = payload.get("source", "Unknown")
                course = payload.get("course_code", "UNKNOWN")
                unit = payload.get("unit", 1)

                key = f"{course}::{source}"
                if key not in docs_summary:
                    docs_summary[key] = {
                        "source": source,
                        "course_code": course,
                        "unit": unit,
                        "chunk_count": 0,
                    }
                docs_summary[key]["chunk_count"] += 1

            if next_offset is None:
                break
            offset = next_offset

        return list(docs_summary.values())