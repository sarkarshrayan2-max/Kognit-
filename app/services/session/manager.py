import json
import time
from typing import Dict, List, Optional
from uuid import UUID

import redis

from app.core.config import settings


class SessionManager:

    def __init__(
        self,
        max_history_messages: int = 12,
        ttl_seconds: int = 60 * 60 * 24,
    ):
        self.max_history_messages = max_history_messages
        self.ttl_seconds = ttl_seconds

        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )

    def _session_key(
        self,
        user_id: UUID | str,
        session_id: str,
    ) -> str:
        return f"kognit:session:{user_id}:{session_id}"

    def _state_key(
        self,
        user_id: UUID | str,
        session_id: str,
    ) -> str:
        return f"kognit:state:{user_id}:{session_id}"

    def add_message(
        self,
        user_id: UUID | str,
        session_id: str,
        role: str,
        content: str,
        course_code: str,
        metadata: Optional[Dict] = None,
    ) -> None:

        key = self._session_key(
            user_id,
            session_id,
        )

        message = {
            "role": role,
            "content": content,
            "course_code": course_code.upper(),
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        self.redis.rpush(
            key,
            json.dumps(message),
        )

        self.redis.ltrim(
            key,
            -self.max_history_messages,
            -1,
        )

        self.redis.expire(
            key,
            self.ttl_seconds,
        )

    def get_context(
        self,
        user_id: UUID | str,
        session_id: str,
        current_course: str,
    ) -> List[Dict[str, str]]:

        key = self._session_key(
            user_id,
            session_id,
        )

        raw_messages = self.redis.lrange(
            key,
            0,
            -1,
        )

        current_course = current_course.upper()

        history: List[Dict[str, str]] = []

        for raw in raw_messages:

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if (
                message.get("course_code", "").upper()
                != current_course
            ):
                continue

            history.append(
                {
                    "role": message.get(
                        "role",
                        "user",
                    ),
                    "content": message.get(
                        "content",
                        "",
                    ),
                }
            )

        return history[
            -self.max_history_messages:
        ]

    def set_state(
        self,
        user_id: UUID | str,
        session_id: str,
        state: Dict,
    ) -> None:

        key = self._state_key(
            user_id,
            session_id,
        )

        self.redis.setex(
            key,
            self.ttl_seconds,
            json.dumps(state),
        )

    def get_state(
        self,
        user_id: UUID | str,
        session_id: str,
    ) -> Optional[Dict]:

        key = self._state_key(
            user_id,
            session_id,
        )

        value = self.redis.get(key)

        if not value:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def clear_session(
        self,
        user_id: UUID | str,
        session_id: str,
    ) -> None:

        self.redis.delete(
            self._session_key(
                user_id,
                session_id,
            ),
            self._state_key(
                user_id,
                session_id,
            ),
        )


session_manager = SessionManager()