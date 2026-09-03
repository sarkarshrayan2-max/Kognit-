from typing import Any, Dict, List, Optional, Tuple
from app.services.search.restricted_web import RestrictedWebSearch

class CRAGEvaluator:
    def __init__(self, high_threshold: float = 0.65, low_threshold: float = 0.20):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.web_searcher = RestrictedWebSearch()

    def evaluate_and_route(
        self,
        query: str,
        local_chunks: List[Dict[str, Any]],
        course_code: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        top_score = local_chunks[0].get("score", 0.0) if local_chunks else 0.0

        if top_score >= self.high_threshold:
            return "CORRECT", local_chunks

        if top_score < self.low_threshold:
            web_results = self.web_searcher.search(
                query=query,
                course_code=course_code,
                max_results=3
            )
            return "INCORRECT", web_results

        web_results = self.web_searcher.search(
            query=query,
            course_code=course_code,
            max_results=2
        )
        blended = local_chunks[:1] + web_results
        return "AMBIGUOUS", blended