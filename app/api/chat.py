import json
import logging
from typing import AsyncIterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest
from app.services.llm.gateway import LLMGateway
from app.services.rag.condenser import QueryCondenser
from app.services.rag.crag import CRAGEvaluator
from app.services.retrieval.fusion import HybridRetriever
from app.services.session.manager import session_manager

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger("kognit.chat")

retriever = HybridRetriever()
crag = CRAGEvaluator()
llm_gateway = LLMGateway()
condenser = QueryCondenser()

@router.post("/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    session_id = payload.session_id or "default_session"

    # 1. Check server memory first; fall back to payload.history from client
    server_history = session_manager.get_context(
        session_id=session_id,
        current_course=payload.course_code
    )
    payload_history = [
        {"role": m.role, "content": m.content}
        for m in (payload.history or [])
    ]
    
    # Unified history resolution
    history = server_history if server_history else payload_history

    logger.info(
        "Session: %s | Course: %s | Prior turns loaded: %d",
        session_id, payload.course_code, len(history)
    )

    # 2. Dynamic JSON intent classification & query condensation (no hardcoded keywords)
    intent, standalone_query = condenser.analyze(
        query=payload.query,
        history=history,
        course_code=payload.course_code
    )
    logger.info("Intent: %s | Raw: '%s' -> Standalone: '%s'", intent, payload.query, standalone_query)

    # Short-circuit conversational filler without querying Qdrant/Tavily
    if intent == "CONVERSATIONAL":
        async def conversational_generator() -> AsyncIterator[str]:
            ack = "Understood! Let me know if you want to explore more examples or dive into another topic."
            session_manager.add_message(session_id, "user", payload.query, payload.course_code)
            session_manager.add_message(session_id, "assistant", ack, payload.course_code)

            meta_event = {
                "type": "metadata",
                "crag_decision": "CONVERSATIONAL",
                "citations": [],
                "model_used": llm_gateway.model_name,
                "standalone_query": payload.query,
            }
            yield f"data: {json.dumps(meta_event)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': ack})}\n\n"

        return StreamingResponse(
            conversational_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 3. Hybrid search (dense + sparse BM25)
    candidates = retriever.search(
        query=standalone_query,
        course_code=payload.course_code,
        top_k=payload.top_k or 3,
    )

    # 4. CRAG validation & routing with course context
    decision, final_context = crag.evaluate_and_route(
        query=standalone_query,
        local_chunks=candidates,
        course_code=payload.course_code,
    )

    citations = [
        {
            "source": c.get("metadata", {}).get("source", "Course Document"),
            "page": c.get("metadata", {}).get("page", "?"),
            "score": round(c.get("score", 0.0), 4),
            "excerpt": c.get("text", "")[:400] + ("..." if len(c.get("text", "")) > 400 else "")
        }
        for c in final_context
    ]

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
                session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=payload.query,
                    course_code=payload.course_code
                )
                session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_text,
                    course_code=payload.course_code,
                    metadata={"citations": citations, "crag_decision": decision}
                )
                logger.info("Persisted Turn for session %s (Length: %d chars)", session_id, len(full_text))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.delete("/session/{session_id}")
def clear_session_endpoint(session_id: str):
    session_manager.clear_session(session_id)
    logger.info("Purged session state for %s", session_id)
    return {"status": "cleared", "session_id": session_id}