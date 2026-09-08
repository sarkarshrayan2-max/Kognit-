import json
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

import redis
import redis.exceptions

from app.core.config import settings

logger = logging.getLogger("kognit.session")


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
        metadata: Optional[Dict[str, Any]] = None,
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

        try:
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
        except redis.exceptions.RedisError as exc:
            logger.warning(
                "Redis unavailable while storing session message: %s",
                exc,
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

        try:
            raw_messages = self.redis.lrange(
                key,
                0,
                -1,
            )
        except redis.exceptions.RedisError as exc:
            logger.warning(
                "Redis unavailable while reading session context: %s",
                exc,
            )
            return []

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
        state: Dict[str, Any],
    ) -> None:
        key = self._state_key(
            user_id,
            session_id,
        )

        try:
            self.redis.setex(
                key,
                self.ttl_seconds,
                json.dumps(
                    state,
                    ensure_ascii=False,
                ),
            )
        except redis.exceptions.RedisError as exc:
            logger.warning(
                "Redis unavailable while storing session state: %s",
                exc,
            )

    def get_state(
        self,
        user_id: UUID | str,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        key = self._state_key(
            user_id,
            session_id,
        )

        try:
            value = self.redis.get(key)
        except redis.exceptions.RedisError as exc:
            logger.warning(
                "Redis unavailable while reading session state: %s",
                exc,
            )
            return None

        if not value:
            return None

        try:
            state = json.loads(value)

            if not isinstance(state, dict):
                return None

            return state

        except json.JSONDecodeError:
            return None

    def clear_session(
        self,
        user_id: UUID | str,
        session_id: str,
    ) -> None:
        try:
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
        except redis.exceptions.RedisError as exc:
            logger.warning(
                "Redis unavailable while clearing session state: %s",
                exc,
            )


session_manager = SessionManager()