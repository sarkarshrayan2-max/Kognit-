import os
from typing import Any, Dict, List, Tuple
from app.services.search.restricted_web import RestrictedWebSearch

# CRAG decision threshold constants
HIGH_CONFIDENCE_THRESHOLD = 0.65
LOW_CONFIDENCE_THRESHOLD = 0.20


class CRAGEvaluator:
    def __init__(self):
        self.web_search = RestrictedWebSearch()

    def evaluate_and_route(
        self, query: str, local_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        top_score = local_chunks[0]["score"] if local_chunks else -1.0

        # Scenario A: High relevance -> strictly use institutional material
        if top_score >= HIGH_CONFIDENCE_THRESHOLD:
            return "CORRECT", local_chunks

        # Scenario B: Moderate relevance -> supplement with restricted web search
        if top_score >= LOW_CONFIDENCE_THRESHOLD:
            web_results = self.web_search.search(query=query, max_results=2)
            merged = local_chunks + web_results
            return "AMBIGUOUS", merged

        # Scenario C: Low/no relevance -> discard local chunks, rely on web search
        web_results = self.web_search.search(query=query, max_results=3)
        return "INCORRECT", web_results