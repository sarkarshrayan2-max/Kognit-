from langgraph.graph import END, START, StateGraph
from app.services.llm.gateway import LLMGateway
from app.services.rag.context import KognitGraphState
from app.services.rag.crag import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)
from typing import Any, Dict, List
from app.services.retrieval.fusion import HybridRetriever
from app.services.search.restricted_web import RestrictedWebSearch


retriever = HybridRetriever()
web_search = RestrictedWebSearch()
llm_gateway = LLMGateway()



def retrieve_node(state: KognitGraphState) -> Dict[str, Any]:
    candidates = retriever.search(
        query=state["query"],
        course_code=state["course_code"],
        top_k=state.get("top_k", 3),
    )
    return {"local_chunks": candidates}



def evaluate_crag_node(state: KognitGraphState) -> Dict[str, Any]:
    local_chunks = state.get("local_chunks", [])
    top_score = local_chunks[0]["score"] if local_chunks else -1.0

    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        return {
            "crag_decision": "CORRECT",
            "final_context": local_chunks,
        }
    elif top_score >= LOW_CONFIDENCE_THRESHOLD:
        return {"crag_decision": "AMBIGUOUS"}
    else:
        return {"crag_decision": "INCORRECT"}



def web_fallback_node(state: KognitGraphState) -> Dict[str, Any]:
    decision = state["crag_decision"]
    query = state["query"]
    local_chunks = state.get("local_chunks", [])

    if decision == "AMBIGUOUS":
        web_results = web_search.search(query=query, max_results=2)
        return {"final_context": local_chunks + web_results}
    else:  
        web_results = web_search.search(query=query, max_results=3)
        return {"final_context": web_results}



def generate_node(state: KognitGraphState) -> Dict[str, Any]:
    context = state.get("final_context", [])
    if not context:
        return {
            "answer": "I could not find sufficient technical details in the course notes or verified engineering archives to answer this accurately.",
            "citations": [],
            "model_used": "none",
        }

    res = llm_gateway.generate_answer(
        query=state["query"],
        retrieved_chunks=context,
    )
    return {
        "answer": res["answer"],
        "citations": res["citations"],
        "model_used": res["model_used"],
    }



def route_crag(state: KognitGraphState) -> str:
    decision = state.get("crag_decision")
    if decision == "CORRECT":
        return "generate"
    return "web_fallback"



workflow = StateGraph(KognitGraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate_crag", evaluate_crag_node)
workflow.add_node("web_fallback", web_fallback_node)
workflow.add_node("generate", generate_node)


workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "evaluate_crag")
workflow.add_conditional_edges(
    "evaluate_crag",
    route_crag,
    {
        "generate": "generate",
        "web_fallback": "web_fallback",
    },
)
workflow.add_edge("web_fallback", "generate")
workflow.add_edge("generate", END)

kognit_graph = workflow.compile()