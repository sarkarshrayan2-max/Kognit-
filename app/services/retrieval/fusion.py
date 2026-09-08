from typing import Any, Dict, List

import torch
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SparseVector,
)
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)

from app.core.config import settings


class HybridRetriever:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        self.collection_name = settings.qdrant_collection

        self.dense_model = SentenceTransformer(
            settings.dense_model,
            device=device,
        )

        self.sparse_model = SparseTextEmbedding(
            model_name=settings.sparse_model,
        )

        self.reranker = CrossEncoder(
            settings.reranker_model,
            device=device,
        )

    @staticmethod
    def _chunk_key(payload: Dict[str, Any], point_id: Any) -> str:
        document_id = payload.get("document_id")
        page = payload.get("page")
        chunk_index = payload.get("chunk_index")

        if document_id is not None and page is not None and chunk_index is not None:
            return f"{document_id}:{page}:{chunk_index}"

        content_hash = payload.get("content_hash")

        if content_hash is not None and page is not None and chunk_index is not None:
            return f"{content_hash}:{page}:{chunk_index}"

        return str(point_id)

    @classmethod
    def _add_candidate(
        cls,
        hit: Any,
        rrf_score: float,
        rrf_scores: Dict[str, float],
        payload_map: Dict[str, Dict[str, Any]],
        chunk_id_map: Dict[str, str],
    ) -> None:
        payload = hit.payload or {}

        chunk_key = cls._chunk_key(
            payload,
            hit.id,
        )

        if chunk_key not in chunk_id_map:
            internal_id = str(hit.id)

            chunk_id_map[chunk_key] = internal_id
            payload_map[internal_id] = payload
            rrf_scores[internal_id] = rrf_score

            return

        internal_id = chunk_id_map[chunk_key]

        rrf_scores[internal_id] = (
            rrf_scores.get(internal_id, 0.0) + rrf_score
        )

    def search(
        self,
        query: str,
        course_code: str,
        top_k: int = 3,
        candidate_limit: int = 10,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        if not course_code:
            return []

        course_filter = Filter(
            must=[
                FieldCondition(
                    key="course_code",
                    match=MatchValue(
                        value=course_code.upper()
                    ),
                )
            ]
        )

        query_dense = self.dense_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        dense_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_dense,
            using="dense",
            query_filter=course_filter,
            limit=candidate_limit,
        )

        dense_results = dense_response.points

        query_sparse = list(
            self.sparse_model.embed([query])
        )[0]

        sparse_vector = SparseVector(
            indices=query_sparse.indices.tolist(),
            values=query_sparse.values.tolist(),
        )

        sparse_response = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vector,
            using="sparse",
            query_filter=course_filter,
            limit=candidate_limit,
        )

        sparse_results = sparse_response.points

        rrf_scores: Dict[str, float] = {}
        payload_map: Dict[str, Dict[str, Any]] = {}
        chunk_id_map: Dict[str, str] = {}

        for rank, hit in enumerate(dense_results):
            score = 1.0 / (
                rrf_k + rank + 1
            )

            self._add_candidate(
                hit=hit,
                rrf_score=score,
                rrf_scores=rrf_scores,
                payload_map=payload_map,
                chunk_id_map=chunk_id_map,
            )

        for rank, hit in enumerate(sparse_results):
            score = 1.0 / (
                rrf_k + rank + 1
            )

            self._add_candidate(
                hit=hit,
                rrf_score=score,
                rrf_scores=rrf_scores,
                payload_map=payload_map,
                chunk_id_map=chunk_id_map,
            )

        sorted_candidates = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:candidate_limit]

        if not sorted_candidates:
            return []

        pairs = []
        valid_candidates = []

        for doc_id, rrf_score in sorted_candidates:
            payload = payload_map.get(doc_id)

            if not payload:
                continue

            text = payload.get("text", "")

            if not text.strip():
                continue

            pairs.append(
                [
                    query,
                    text,
                ]
            )

            valid_candidates.append(
                (
                    doc_id,
                    rrf_score,
                )
            )

        if not pairs:
            return []

        rerank_scores = self.reranker.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        reranked_results: List[Dict[str, Any]] = []

        for index, (doc_id, rrf_score) in enumerate(
            valid_candidates
        ):
            payload = payload_map[doc_id]

            rerank_score = float(
                rerank_scores[index]
            )

            metadata = {
                key: value
                for key, value in payload.items()
                if key != "text"
            }

            metadata["retrieval_score"] = float(
                rrf_score
            )

            metadata["rerank_score"] = rerank_score

            reranked_results.append(
                {
                    "score": rerank_score,
                    "text": payload["text"],
                    "metadata": metadata,
                }
            )

        reranked_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return reranked_results[:top_k]