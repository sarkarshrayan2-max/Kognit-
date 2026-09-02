import os
from typing import Any, Dict, List, Tuple
from app.services.search.restricted_web import RestrictedWebSearch


HIGH_CONFIDENCE_THRESHOLD = 0.65
LOW_CONFIDENCE_THRESHOLD = 0.20


class CRAGEvaluator:
    def __init__(self):
        self.web_search = RestrictedWebSearch()

    def evaluate_and_route(
        self, query: str, local_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        top_score = local_chunks[0]["score"] if local_chunks else -1.0

        
        if top_score >= HIGH_CONFIDENCE_THRESHOLD:
            return "CORRECT", local_chunks

        
        if top_score >= LOW_CONFIDENCE_THRESHOLD:
            web_results = self.web_search.search(query=query, max_results=2)
            merged = local_chunks + web_results
            return "AMBIGUOUS", merged

        
        web_results = self.web_search.search(query=query, max_results=3)
        return "INCORRECT", web_results