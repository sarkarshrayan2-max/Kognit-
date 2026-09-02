import uuid
from pathlib import Path
from typing import Any, Dict, List

import pymupdf 
from fastembed import SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
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
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
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

        
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)