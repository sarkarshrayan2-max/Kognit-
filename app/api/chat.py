import json
import logging
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.graph.workflow import (
    kognit_graph,
    llm_gateway,
)
from app.models.user import User
from app.schemas.chat import ChatRequest
from app.services.chat.persistence import chat_persistence
from app.services.memory.extractor import memory_extractor
from app.services.memory.long_term import long_term_memory
from app.services.session.manager import session_manager

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

logger = logging.getLogger(
    "kognit.chat"
)


def sse_event(
    payload: Dict[str, Any],
) -> str:
    return (
        f"data: "
        f"{json.dumps(payload, ensure_ascii=False)}"
        f"\n\n"
    )


@router.post("/stream")
async def chat_stream_endpoint(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = (
        payload.session_id
        or "default_session"
    )

    conversation = (
        chat_persistence.get_or_create_conversation(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
            course_code=payload.course_code,
        )
    )

    server_history = (
        session_manager.get_context(
            current_user.id,
            session_id,
            payload.course_code,
        )
    )

    if server_history:
        history = server_history
    else:
        history = (
            chat_persistence.get_history(
                db=db,
                conversation_id=conversation.id,
                limit=12,
            )
        )

    if not history:
        history = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in (payload.history or [])
        ]

    initial_state = {
        "query": payload.query,
        "course_code": payload.course_code,
        "history": history,
        "top_k": payload.top_k or 3,
    }

    async def event_generator() -> AsyncIterator[str]:
        accumulated_answer = ""
        metadata_sent = False
        standalone_query = payload.query
        response_type = "TECHNICAL"
        decision = "UNKNOWN"
        citations = []

        try:
            async for chunk in (
                kognit_graph.astream(
                    initial_state,
                    stream_mode=[
                        "custom",
                        "updates",
                    ],
                    version="v2",
                )
            ):
                chunk_type = chunk.get(
                    "type"
                )

                if chunk_type == "custom":
                    data = chunk.get(
                        "data",
                        {},
                    )

                    if not isinstance(
                        data,
                        dict,
                    ):
                        continue

                    if data.get(
                        "type"
                    ) != "token":
                        continue

                    content = data.get(
                        "content",
                        "",
                    )

                    if not content:
                        continue

                    accumulated_answer += content

                    yield sse_event(
                        {
                            "type": "token",
                            "content": content,
                        }
                    )

                    continue

                if chunk_type != "updates":
                    continue

                update_data = chunk.get(
                    "data",
                    {},
                )

                if not isinstance(
                    update_data,
                    dict,
                ):
                    continue

                for (
                    node_name,
                    node_update,
                ) in update_data.items():

                    if not isinstance(
                        node_update,
                        dict,
                    ):
                        continue

                    if node_name == "condenser":
                        standalone_query = (
                            node_update.get(
                                "standalone_query",
                                standalone_query,
                            )
                        )
                        continue

                    if node_name == "crag":
                        decision = (
                            node_update.get(
                                "crag_decision",
                                "UNKNOWN",
                            )
                        )

                        citations = (
                            node_update.get(
                                "citations",
                                [],
                            )
                        )

                        response_type = (
                            node_update.get(
                                "response_type",
                                "TECHNICAL",
                            )
                        )

                        if not metadata_sent:
                            metadata_sent = True

                            yield sse_event(
                                {
                                    "type": "metadata",
                                    "crag_decision": decision,
                                    "citations": citations,
                                    "model_used": (
                                        llm_gateway.model_name
                                    ),
                                    "standalone_query": (
                                        standalone_query
                                    ),
                                    "response_type": (
                                        response_type
                                    ),
                                }
                            )

                        continue

                    if (
                        node_name
                        == "conversational_handler"
                    ):
                        response_type = (
                            node_update.get(
                                "response_type",
                                "CONVERSATIONAL",
                            )
                        )

                        decision = (
                            node_update.get(
                                "crag_decision",
                                "CONVERSATIONAL",
                            )
                        )

                        citations = (
                            node_update.get(
                                "citations",
                                [],
                            )
                        )

                        answer = (
                            node_update.get(
                                "answer",
                                "",
                            )
                        )

                        if not metadata_sent:
                            metadata_sent = True

                            yield sse_event(
                                {
                                    "type": "metadata",
                                    "crag_decision": decision,
                                    "citations": citations,
                                    "model_used": (
                                        llm_gateway.model_name
                                    ),
                                    "standalone_query": (
                                        standalone_query
                                    ),
                                    "response_type": (
                                        response_type
                                    ),
                                }
                            )

                        if (
                            answer
                            and not accumulated_answer
                        ):
                            accumulated_answer = answer

                            yield sse_event(
                                {
                                    "type": "token",
                                    "content": answer,
                                }
                            )

                        continue

                    if node_name == "generator":
                        final_answer = (
                            node_update.get(
                                "answer",
                                "",
                            )
                        )

                        if (
                            final_answer
                            and not accumulated_answer
                        ):
                            accumulated_answer = (
                                final_answer
                            )

                            yield sse_event(
                                {
                                    "type": "token",
                                    "content": final_answer,
                                }
                            )

            if accumulated_answer:
                chat_persistence.save_message(
                    db=db,
                    conversation_id=conversation.id,
                    role="user",
                    content=payload.query,
                )

                chat_persistence.save_message(
                    db=db,
                    conversation_id=conversation.id,
                    role="assistant",
                    content=accumulated_answer,
                    metadata={
                        "citations": citations,
                        "crag_decision": decision,
                        "response_type": response_type,
                        "standalone_query": (
                            standalone_query
                        ),
                    },
                )

                session_manager.add_message(
                    current_user.id,
                    session_id,
                    "user",
                    payload.query,
                    payload.course_code,
                )

                session_manager.add_message(
                    current_user.id,
                    session_id,
                    "assistant",
                    accumulated_answer,
                    payload.course_code,
                    metadata={
                        "citations": citations,
                        "crag_decision": decision,
                        "response_type": response_type,
                    },
                )

                extracted_memories = (
                    memory_extractor.extract(
                        payload.query
                    )
                )

                for memory in extracted_memories:
                    existing_memory = (
                        long_term_memory.get_memory(
                            db=db,
                            user_id=current_user.id,
                            memory_key=memory.memory_key,
                        )
                    )

                    if existing_memory:
                        existing_memory.memory_value = (
                            memory.memory_value
                        )
                        existing_memory.memory_type = (
                            memory.memory_type
                        )
                        existing_memory.importance = (
                            memory.importance
                        )

                        db.commit()
                        db.refresh(
                            existing_memory
                        )

                    else:
                        long_term_memory.save_memory(
                            db=db,
                            user_id=current_user.id,
                            memory_key=memory.memory_key,
                            memory_value=memory.memory_value,
                            memory_type=memory.memory_type,
                            importance=memory.importance,
                        )

            yield sse_event(
                {
                    "type": "done"
                }
            )

        except Exception:
            logger.exception(
                "LangGraph streaming failed"
            )

            yield sse_event(
                {
                    "type": "error",
                    "content": (
                        "An internal error occurred "
                        "while processing your request."
                    ),
                }
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete(
    "/session/{session_id}"
)
def clear_session_endpoint(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    session_manager.clear_session(
        current_user.id,
        session_id,
    )

    return {
        "status": "cleared",
        "session_id": session_id,
        "user_id": str(current_user.id),
    }