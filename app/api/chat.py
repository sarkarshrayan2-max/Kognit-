from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag.retriever import kognit_graph

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    
    initial_state = {
        "query": payload.query,
        "course_code": payload.course_code,
        "top_k": payload.top_k or 3,
        "local_chunks": [],
        "final_context": [],
        "crag_decision": "PENDING",
        "answer": "",
        "citations": [],
        "model_used": "",
    }

    try:
        
        result = kognit_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

    return ChatResponse(
        answer=result["answer"],
        crag_decision=result["crag_decision"],
        citations=result["citations"],
        model_used=result["model_used"],
    )