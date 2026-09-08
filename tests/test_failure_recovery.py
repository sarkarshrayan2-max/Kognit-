from unittest.mock import Mock

import redis

from app.services.session.manager import SessionManager


def test_redis_failure_get_context_returns_empty():
    manager = SessionManager()

    manager.redis = Mock()
    manager.redis.lrange.side_effect = redis.exceptions.ConnectionError(
        "Redis unavailable"
    )

    result = manager.get_context(
        user_id="test-user",
        session_id="test-session",
        current_course="DBMS",
    )

    assert result == []


def test_redis_failure_get_state_returns_none():
    manager = SessionManager()

    manager.redis = Mock()
    manager.redis.get.side_effect = redis.exceptions.ConnectionError(
        "Redis unavailable"
    )

    result = manager.get_state(
        user_id="test-user",
        session_id="test-session",
    )

    assert result is None


def test_redis_failure_add_message_does_not_raise():
    manager = SessionManager()

    manager.redis = Mock()
    manager.redis.rpush.side_effect = redis.exceptions.ConnectionError(
        "Redis unavailable"
    )

    manager.add_message(
        user_id="test-user",
        session_id="test-session",
        role="user",
        content="TEST",
        course_code="DBMS",
    )


def test_redis_failure_set_state_does_not_raise():
    manager = SessionManager()

    manager.redis = Mock()
    manager.redis.setex.side_effect = redis.exceptions.ConnectionError(
        "Redis unavailable"
    )

    manager.set_state(
        user_id="test-user",
        session_id="test-session",
        state={"test": True},
    )