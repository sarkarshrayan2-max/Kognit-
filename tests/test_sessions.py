from app.services.session.manager import SessionManager


def test_session_message_roundtrip():
    manager = SessionManager()

    user_id = "test-user"
    session_id = "test-session"

    manager.redis.delete(
        manager._session_key(user_id, session_id),
        manager._state_key(user_id, session_id),
    )

    manager.add_message(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content="Hello",
        course_code="DBMS",
    )

    manager.add_message(
        user_id=user_id,
        session_id=session_id,
        role="assistant",
        content="Hello. How can I help?",
        course_code="DBMS",
    )

    context = manager.get_context(
        user_id=user_id,
        session_id=session_id,
        current_course="DBMS",
    )

    assert len(context) == 2
    assert context[0]["role"] == "user"
    assert context[0]["content"] == "Hello"
    assert context[1]["role"] == "assistant"
    assert context[1]["content"] == "Hello. How can I help?"

    manager.clear_session(user_id, session_id)


def test_session_state_roundtrip():
    manager = SessionManager()

    user_id = "test-user"
    session_id = "state-test"

    manager.clear_session(user_id, session_id)

    state = {
        "query": "What is normalization?",
        "response_type": "TECHNICAL",
        "crag_decision": "CORRECT",
    }

    manager.set_state(
        user_id=user_id,
        session_id=session_id,
        state=state,
    )

    result = manager.get_state(
        user_id=user_id,
        session_id=session_id,
    )

    assert result == state

    manager.clear_session(user_id, session_id)


def test_clear_session_removes_messages_and_state():
    manager = SessionManager()

    user_id = "test-user"
    session_id = "delete-test"

    manager.add_message(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content="Delete me",
        course_code="DBMS",
    )

    manager.set_state(
        user_id=user_id,
        session_id=session_id,
        state={"test": True},
    )

    manager.clear_session(user_id, session_id)

    context = manager.get_context(
        user_id=user_id,
        session_id=session_id,
        current_course="DBMS",
    )

    state = manager.get_state(
        user_id=user_id,
        session_id=session_id,
    )

    assert context == []
    assert state is None


def test_sessions_are_isolated_by_session_id():
    manager = SessionManager()

    user_id = "test-user"

    session_a = "session-a"
    session_b = "session-b"

    manager.clear_session(user_id, session_a)
    manager.clear_session(user_id, session_b)

    manager.add_message(
        user_id=user_id,
        session_id=session_a,
        role="user",
        content="Message A",
        course_code="DBMS",
    )

    manager.add_message(
        user_id=user_id,
        session_id=session_b,
        role="user",
        content="Message B",
        course_code="DBMS",
    )

    context_a = manager.get_context(
        user_id=user_id,
        session_id=session_a,
        current_course="DBMS",
    )

    context_b = manager.get_context(
        user_id=user_id,
        session_id=session_b,
        current_course="DBMS",
    )

    assert len(context_a) == 1
    assert context_a[0]["content"] == "Message A"

    assert len(context_b) == 1
    assert context_b[0]["content"] == "Message B"

    manager.clear_session(user_id, session_a)
    manager.clear_session(user_id, session_b)