from typing import Any, Dict, List, Optional, Tuple

from app.services.search.restricted_web import RestrictedWebSearch


class CRAGEvaluator:
    """
    CRAG evaluator for KOGNIT.

    Routing:
        - CORRECT   -> strong course-document evidence
        - AMBIGUOUS -> some course evidence, but not strong enough
        - INCORRECT -> weak/no course evidence, optionally use web
    """

    def __init__(
        self,
        high_threshold: float = 0.65,
        low_threshold: float = 0.20,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.web_searcher = RestrictedWebSearch()

    @staticmethod
    def _course_chunks(
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        normalized = []

        for chunk in chunks:
            item = dict(chunk)

            metadata = dict(item.get("metadata", {}))
            metadata["source_type"] = "course"

            item["metadata"] = metadata
            normalized.append(item)

        return normalized

    @staticmethod
    def _web_chunks(
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        normalized = []

        for chunk in chunks:
            item = dict(chunk)

            metadata = dict(item.get("metadata", {}))
            metadata["source_type"] = "web"

            item["metadata"] = metadata
            normalized.append(item)

        return normalized

    @staticmethod
    def _score(chunk: Dict[str, Any]) -> float:
        """
        Safely extract reranker/retrieval score.
        """

        if "rerank_score" in chunk:
            return float(chunk["rerank_score"])

        return float(chunk.get("score", 0.0))

    def evaluate_and_route(
        self,
        query: str,
        local_chunks: List[Dict[str, Any]],
        course_code: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:

        

        if not local_chunks:

            web_results = self.web_searcher.search(
                query=query,
                course_code=course_code,
                max_results=3,
            )

            if web_results:
                return "INCORRECT", self._web_chunks(web_results)

            return "INCORRECT", []

    

        course_chunks = self._course_chunks(local_chunks)

        

        course_chunks.sort(
            key=self._score,
            reverse=True,
        )

        top_score = self._score(course_chunks[0])


        if top_score >= self.high_threshold:

            return "CORRECT", course_chunks[:3]



        if top_score >= self.low_threshold:

            web_results = self.web_searcher.search(
                query=query,
                course_code=course_code,
                max_results=2,
            )

            routed_context = course_chunks[:2]

            if web_results:
                routed_context.extend(
                    self._web_chunks(web_results)
                )

            return "AMBIGUOUS", routed_context

        

        web_results = self.web_searcher.search(
            query=query,
            course_code=course_code,
            max_results=3,
        )

        if web_results:

            return "INCORRECT", self._web_chunks(web_results)

        return "INCORRECT", []