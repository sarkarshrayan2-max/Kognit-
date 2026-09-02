from typing import Any, Dict, List
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector
from sentence_transformers import CrossEncoder, SentenceTransformer

COLLECTION_NAME = "ecs_knowledge_base"


class HybridRetriever:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.dense_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.reranker = CrossEncoder("BAAI/bge-reranker-large")

    def search(
        self,
        query: str,
        course_code: str,
        top_k: int = 3,
        candidate_limit: int = 25,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        # Enforce metadata pre-filtering by course
        course_filter = Filter(
            must=[
                FieldCondition(
                    key="course_code",
                    match=MatchValue(value=course_code.upper()),
                )
            ]
        )

        # 1. Dense Search using query_points
        query_dense = self.dense_model.encode(
            query, normalize_embeddings=True
        ).tolist()
        
        dense_response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_dense,
            using="dense",
            query_filter=course_filter,
            limit=candidate_limit,
        )
        dense_results = dense_response.points

        # 2. Sparse (BM25) Search using query_points
        query_sparse = list(self.sparse_model.embed([query]))[0]
        sparse_vector = SparseVector(
            indices=query_sparse.indices.tolist(),
            values=query_sparse.values.tolist(),
        )

        sparse_response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=sparse_vector,
            using="sparse",
            query_filter=course_filter,
            limit=candidate_limit,
        )
        sparse_results = sparse_response.points

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        payload_map: Dict[str, Any] = {}

        for rank, hit in enumerate(dense_results):
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            payload_map[hit.id] = hit.payload

        for rank, hit in enumerate(sparse_results):
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            payload_map[hit.id] = hit.payload

        sorted_candidates = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:candidate_limit]

        if not sorted_candidates:
            return []

        # 4. Cross-Encoder Reranking
        pairs = [
            [query, payload_map[doc_id]["text"]]
            for doc_id, _ in sorted_candidates
        ]
        rerank_scores = self.reranker.predict(pairs)

        reranked_results = []
        for i, (doc_id, _) in enumerate(sorted_candidates):
            reranked_results.append(
                {
                    "score": float(rerank_scores[i]),
                    "text": payload_map[doc_id]["text"],
                    "metadata": {
                        k: v
                        for k, v in payload_map[doc_id].items()
                        if k != "text"
                    },
                }
            )

        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        return reranked_results[:top_k]