from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class MessageItem(BaseModel):
    role: str  
    content: str

class ChatRequest(BaseModel):
    query: str
    course_code: str
    session_id: Optional[str] = Field(default="default_session")
    history: Optional[List[MessageItem]] = Field(default_factory=list)
    top_k: Optional[int] = Field(default=3)

class Citation(BaseModel):
    source: Optional[str] = "Unknown"
    page: Optional[Any] = "?"
    score: float
    excerpt: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    crag_decision: str
    citations: List[Citation]
    model_used: str
    standalone_query: Optional[str] = None