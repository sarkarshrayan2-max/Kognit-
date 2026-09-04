import json
import logging
from typing import AsyncIterator, Dict, Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.graph.workflow import (
    kognit_graph,
    llm_gateway,
)

from app.schemas.chat import (
    ChatRequest,
)

from app.services.session.manager import (
    session_manager,
)


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
):

    session_id = (
        payload.session_id
        or "default_session"
    )

    

    server_history = (
        session_manager.get_context(
            session_id=session_id,
            current_course=
                payload.course_code,
        )
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

        "course_code":
            payload.course_code,

        "history":
            history,

        "top_k":
            payload.top_k or 3,
    }

    

    async def event_generator(
    ) -> AsyncIterator[str]:

        accumulated_answer = ""

        metadata_sent = False

        standalone_query = (
            payload.query
        )

        response_type = (
            "TECHNICAL"
        )

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

                chunk_type = (
                    chunk.get(
                        "type"
                    )
                )

                

                if (
                    chunk_type
                    == "custom"
                ):

                    data = chunk.get(
                        "data",
                        {},
                    )

                    if not isinstance(
                        data,
                        dict,
                    ):
                        continue

                    if (
                        data.get("type")
                        != "token"
                    ):
                        continue

                    content = data.get(
                        "content",
                        "",
                    )

                    if not content:
                        continue

                    accumulated_answer += (
                        content
                    )

                    yield sse_event(
                        {
                            "type":
                                "token",

                            "content":
                                content,
                        }
                    )

                    continue

                
                if (
                    chunk_type
                    != "updates"
                ):
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

                

                    if (
                        node_name
                        == "condenser"
                    ):

                        standalone_query = (
                            node_update.get(
                                "standalone_query",
                                standalone_query,
                            )
                        )

                        continue

                    

                    if (
                        node_name
                        == "crag"
                    ):

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
                                    "type":
                                        "metadata",

                                    "crag_decision":
                                        decision,

                                    "citations":
                                        citations,

                                    "model_used":
                                        llm_gateway.model_name,

                                    "standalone_query":
                                        standalone_query,

                                    "response_type":
                                        response_type,
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
                                    "type":
                                        "metadata",

                                    "crag_decision":
                                        decision,

                                    "citations":
                                        citations,

                                    "model_used":
                                        llm_gateway.model_name,

                                    "standalone_query":
                                        standalone_query,

                                    "response_type":
                                        response_type,
                                }
                            )

                        
                        if (
                            answer
                            and not accumulated_answer
                        ):

                            accumulated_answer = (
                                answer
                            )

                            yield sse_event(
                                {
                                    "type":
                                        "token",

                                    "content":
                                        answer,
                                }
                            )

                        continue

                    

                    if (
                        node_name
                        == "generator"
                    ):

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
                                    "type":
                                        "token",

                                    "content":
                                        final_answer,
                                }
                            )

            

            if accumulated_answer:

                session_manager.add_message(
                    session_id=session_id,

                    role="user",

                    content=payload.query,

                    course_code=
                        payload.course_code,
                )

                session_manager.add_message(
                    session_id=session_id,

                    role="assistant",

                    content=
                        accumulated_answer,

                    course_code=
                        payload.course_code,

                    metadata={
                        "citations":
                            citations,

                        "crag_decision":
                            decision,
                    },
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
                        "An internal error "
                        "occurred while "
                        "processing your request."
                    ),
                }
            )


    return StreamingResponse(
        event_generator(),

        media_type=
            "text/event-stream",

        headers={
            "Cache-Control":
                "no-cache, no-transform",

            "Connection":
                "keep-alive",

            "X-Accel-Buffering":
                "no",

            "X-Content-Type-Options":
                "nosniff",
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
        "status":
            "cleared",

        "session_id":
            session_id,
    }