import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.graph.workflow import kognit_graph, llm_gateway
from app.schemas.chat import ChatRequest
from app.services.session.manager import session_manager


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

logger = logging.getLogger("kognit.chat")


@router.post("/stream")
async def chat_stream_endpoint(
    payload: ChatRequest,
):
    session_id = (
        payload.session_id
        or "default_session"
    )


    server_history = session_manager.get_context(
        session_id=session_id,
        current_course=payload.course_code,
    )


    payload_history = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in (
            payload.history or []
        )
    ]

    history = (
        server_history
        if server_history
        else payload_history
    )

    initial_state = {
        "query": payload.query,
        "course_code": payload.course_code,
        "history": history,
        "top_k": payload.top_k or 3,
    }

    try:
        graph_output = await kognit_graph.ainvoke(
            initial_state
        )

    except Exception as exc:
        logger.exception(
            "LangGraph execution failed"
        )

        async def error_generator():
            error_event = {
                "type": "error",
                "content": (
                    "An internal error occurred "
                    "while processing your request."
                ),
            }

            yield (
                f"data: "
                f"{json.dumps(error_event)}"
                f"\n\n"
            )

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
        )

    response_type = graph_output.get(
        "response_type",
        "TECHNICAL",
    )

    answer = graph_output.get(
        "answer",
        "",
    )

    decision = graph_output.get(
        "crag_decision",
        "UNKNOWN",
    )

    standalone_query = graph_output.get(
        "standalone_query",
        payload.query,
    )

    citations = graph_output.get(
        "citations",
        [],
    )


    async def event_generator() -> AsyncIterator[str]:

        try:


            metadata_event = {
                "type": "metadata",
                "crag_decision": decision,
                "citations": citations,
                "model_used": (
                    llm_gateway.model_name
                ),
                "standalone_query": (
                    standalone_query
                ),
                "response_type": response_type,
            }

            yield (
                f"data: "
                f"{json.dumps(metadata_event)}"
                f"\n\n"
            )


            if answer:

                token_event = {
                    "type": "token",
                    "content": answer,
                }

                yield (
                    f"data: "
                    f"{json.dumps(token_event)}"
                    f"\n\n"
                )
            yield (
                f"data: "
                f"{json.dumps({'type': 'done'})}"
                f"\n\n"
            )

            session_manager.add_message(
                session_id=session_id,
                role="user",
                content=payload.query,
                course_code=payload.course_code,
            )

            session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=answer,
                course_code=payload.course_code,
                metadata={
                    "citations": citations,
                    "crag_decision": decision,
                },
            )

        except Exception as exc:

            logger.exception(
                "SSE response failed"
            )

            error_event = {
                "type": "error",
                "content": (
                    "An error occurred while "
                    "sending the response."
                ),
            }

            yield (
                f"data: "
                f"{json.dumps(error_event)}"
                f"\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@router.delete(
    "/session/{session_id}"
)
def clear_session_endpoint(
    session_id: str,
):
    session_manager.clear_session(
        session_id
    )
    return {
        "status": "cleared",
        "session_id": session_id,
    }