from typing import Any, List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    course_code: str
    top_k: Optional[int] = 3


class Citation(BaseModel):
    source: str
    page: Any
    score: float


class ChatResponse(BaseModel):
    answer: str
    crag_decision: str
    citations: List[Citation]
    model_used: str