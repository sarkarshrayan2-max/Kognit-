from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class KognitGraphState(TypedDict):
    query: str
    course_code: str
    top_k: int
    local_chunks: List[Dict[str, Any]]
    final_context: List[Dict[str, Any]]
    crag_decision: str
    answer: str
    citations: List[Dict[str, Any]]
    model_used: str