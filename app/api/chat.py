import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest
from app.services.llm.gateway import LLMGateway
from app.services.rag.crag import CRAGEvaluator
from app.services.retrieval.fusion import HybridRetriever

router = APIRouter(prefix="/chat", tags=["Chat"])

retriever = HybridRetriever()
crag = CRAGEvaluator()
llm_gateway = LLMGateway()


@router.post("/stream")
def chat_stream_endpoint(payload: ChatRequest):
    # 1. Retrieve candidates
    candidates = retriever.search(
        query=payload.query,
        course_code=payload.course_code,
        top_k=payload.top_k or 3,
    )

    # 2. Evaluate with CRAG
    decision, final_context = crag.evaluate_and_route(
        query=payload.query,
        local_chunks=candidates,
    )

    citations = [
        {
            "source": c.get("metadata", {}).get("source"),
            "page": c.get("metadata", {}).get("page"),
            "score": round(c.get("score", 0.0), 4),
        }
        for c in final_context
    ]

    # Standard synchronous generator for StreamingResponse
    def event_generator():
        try:
            # Step A: Send metadata event first
            meta_event = {
                "type": "metadata",
                "crag_decision": decision,
                "citations": citations,
                "model_used": llm_gateway.model_name,
            }
            yield f"data: {json.dumps(meta_event)}\n\n"

            if not final_context:
                error_event = {
                    "type": "token",
                    "content": "No sufficient material found in local documents or external sources.",
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                return

            # Step B: Stream tokens safely
            for token in llm_gateway.stream_answer(payload.query, final_context):
                if token:
                    token_event = {"type": "token", "content": token}
                    yield f"data: {json.dumps(token_event)}\n\n"

        except Exception as err:
            err_event = {
                "type": "token",
                "content": f"\n\n[Generation error: {str(err)}]",
            }
            yield f"data: {json.dumps(err_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )