from typing import List, Dict, Optional
import time
from collections import defaultdict
from pydantic import BaseModel, Field

class StoredMessage(BaseModel):
    role: str
    content: str
    course_code: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Optional[Dict] = None

class SessionManager:
    """
    Manages active conversation sessions. 
    Can be swapped seamlessly with Redis or SQLAlchemy session stores.
    """
    def __init__(self, max_history_turns: int = 6):
        self._storage: Dict[str, List[StoredMessage]] = defaultdict(list)
        self.max_history_turns = max_history_turns

    def add_message(self, session_id: str, role: str, content: str, course_code: str, metadata: Optional[Dict] = None):
        msg = StoredMessage(
            role=role,
            content=content,
            course_code=course_code,
            metadata=metadata or {}
        )
        self._storage[session_id].append(msg)

    def get_context(self, session_id: str, current_course: str) -> List[Dict[str, str]]:
        """
        Retrieves history strictly isolated to the specified course_code
        to avoid topic bleed across different engineering subjects.
        """
        all_msgs = self._storage.get(session_id, [])
        filtered = [
            {"role": m.role, "content": m.content}
            for m in all_msgs
            if m.course_code == current_course
        ]
        return filtered[-self.max_history_turns:]

    def clear_session(self, session_id: str):
        if session_id in self._storage:
            del self._storage[session_id]


session_manager = SessionManager()