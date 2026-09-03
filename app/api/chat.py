import json
import logging
from typing import AsyncIterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest
from app.graph.workflow import kognit_graph, llm_gateway
from app.services.session.manager import session_manager

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger("kognit.chat")

@router.post("/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    session_id = payload.session_id or "default_session"


    server_history = session_manager.get_context(
        session_id=session_id,
        current_course=payload.course_code
    )
    payload_history = [
        {"role": m.role, "content": m.content}
        for m in (payload.history or [])
    ]
    history = server_history if server_history else payload_history

    initial_state = {
        "query": payload.query,
        "course_code": payload.course_code,
        "history": history,
        "top_k": payload.top_k or 3,
    }

    
    graph_output = await kognit_graph.ainvoke(initial_state)

    response_type = graph_output.get("response_type", "TECHNICAL")
    standalone_query = graph_output.get("standalone_query", payload.query)
    decision = graph_output.get("crag_decision", "UNKNOWN")
    final_context = graph_output.get("final_context", [])
    citations = graph_output.get("citations", [])
    static_answer = graph_output.get("answer")

    
    if response_type == "CONVERSATIONAL":
        async def conversational_generator() -> AsyncIterator[str]:
            session_manager.add_message(session_id, "user", payload.query, payload.course_code)
            session_manager.add_message(session_id, "assistant", static_answer, payload.course_code)

            meta_event = {
                "type": "metadata",
                "crag_decision": decision,
                "citations": [],
                "model_used": llm_gateway.model_name,
                "standalone_query": payload.query,
            }
            yield f"data: {json.dumps(meta_event)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': static_answer})}\n\n"

        return StreamingResponse(
            conversational_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    
    async def event_generator() -> AsyncIterator[str]:
        accumulated_answer = []
        try:
            meta_event = {
                "type": "metadata",
                "crag_decision": decision,
                "citations": citations,
                "model_used": llm_gateway.model_name,
                "standalone_query": standalone_query,
            }
            yield f"data: {json.dumps(meta_event)}\n\n"

            if not final_context:
                fallback_msg = "No sufficient material found in local documents or external sources."
                accumulated_answer.append(fallback_msg)
                yield f"data: {json.dumps({'type': 'token', 'content': fallback_msg})}\n\n"
                return

            for token in llm_gateway.stream_answer(
                query=payload.query,
                retrieved_chunks=final_context,
                history=history,
                crag_decision=decision,
            ):
                if token:
                    accumulated_answer.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        except Exception as err:
            logger.error("Streaming failure: %s", str(err), exc_info=True)
            err_msg = f"\n\n[Generation error: {str(err)}]"
            accumulated_answer.append(err_msg)
            yield f"data: {json.dumps({'type': 'token', 'content': err_msg})}\n\n"

        finally:
            if accumulated_answer:
                full_text = "".join(accumulated_answer)
                session_manager.add_message(session_id, "user", payload.query, payload.course_code)
                session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_text,
                    course_code=payload.course_code,
                    metadata={"citations": citations, "crag_decision": decision}
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.delete("/session/{session_id}")
def clear_session_endpoint(session_id: str):
    session_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}