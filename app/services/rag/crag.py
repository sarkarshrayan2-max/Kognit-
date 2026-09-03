from typing import Any, Dict, List, Optional, Tuple

from app.services.search.restricted_web import RestrictedWebSearch


class CRAGEvaluator:
    """
    Corrective RAG evaluator.

    Routing:
        HIGH confidence  -> course documents only
        MEDIUM confidence -> course + supplementary web
        LOW confidence    -> web fallback

    Important:
    Reranker scores are NOT probabilities.
    Thresholds should eventually be tuned using an evaluation dataset.
    """

    def __init__(
        self,
        high_threshold: float = 0.65,
        low_threshold: float = 0.20,
        min_score_gap: float = 0.05,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.min_score_gap = min_score_gap

        self.web_searcher = RestrictedWebSearch()

    def _course_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Mark retrieved chunks as course sources.
        """

        normalized = []

        for chunk in chunks:
            item = dict(chunk)

            metadata = dict(item.get("metadata", {}))

            metadata["source_type"] = "course"

            item["metadata"] = metadata

            normalized.append(item)

        return normalized

    def _web_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Mark external search results as web sources.
        """

        normalized = []

        for chunk in chunks:
            item = dict(chunk)

            metadata = dict(item.get("metadata", {}))

            metadata["source_type"] = "web"

            item["metadata"] = metadata

            normalized.append(item)

        return normalized

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

            web_results = self._web_chunks(web_results)

            return "INCORRECT", web_results

        
        course_chunks = self._course_chunks(local_chunks)

        

        top_score = course_chunks[0].get("score", 0.0)

        second_score = (
            course_chunks[1].get("score", 0.0)
            if len(course_chunks) > 1
            else 0.0
        )

        score_gap = top_score - second_score

        

        if (
            top_score >= self.high_threshold
            and score_gap >= self.min_score_gap
        ):
            return "CORRECT", course_chunks

        

        if top_score >= self.low_threshold:

            web_results = self.web_searcher.search(
                query=query,
                course_code=course_code,
                max_results=2,
            )

            web_results = self._web_chunks(web_results)

            blended = course_chunks[:2] + web_results

            return "AMBIGUOUS", blended

        

        web_results = self.web_searcher.search(
            query=query,
            course_code=course_code,
            max_results=3,
        )

        web_results = self._web_chunks(web_results)

        return "INCORRECT", web_results