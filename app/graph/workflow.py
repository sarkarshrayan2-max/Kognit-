import logging
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from app.services.rag.condenser import QueryCondenser
from app.services.rag.crag import CRAGEvaluator
from app.services.retrieval.fusion import HybridRetriever
from app.services.llm.gateway import LLMGateway

logger = logging.getLogger("kognit.workflow")

retriever = HybridRetriever()
crag_evaluator = CRAGEvaluator()
condenser = QueryCondenser()
llm_gateway = LLMGateway()

class GraphState(TypedDict):
    query: str
    course_code: str
    history: List[Dict[str, str]]
    top_k: int
    intent: str
    standalone_query: str
    response_type: str  
    answer: Optional[str]  
    local_chunks: List[Dict[str, Any]]
    final_context: List[Dict[str, Any]]
    crag_decision: str
    citations: List[Dict[str, Any]]

def condense_node(state: GraphState) -> Dict[str, Any]:
    intent, standalone = condenser.analyze(
        query=state["query"],
        history=state.get("history", []),
        course_code=state["course_code"]
    )
    return {
        "intent": intent, 
        "standalone_query": standalone
    }

def conversational_node(state: GraphState) -> Dict[str, Any]:
    ack = "Understood! Let me know if you want to explore more examples or dive into another topic."
    return {
        "response_type": "CONVERSATIONAL",
        "answer": ack,
        "crag_decision": "CONVERSATIONAL",
        "final_context": [],
        "citations": []
    }

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    k = state.get("top_k", 3)
    chunks = retriever.search(
        query=state["standalone_query"],
        course_code=state["course_code"],
        top_k=k
    )
    return {"local_chunks": chunks}

def crag_eval_node(state: GraphState) -> Dict[str, Any]:
    decision, routed_context = crag_evaluator.evaluate_and_route(
        query=state["standalone_query"],
        local_chunks=state.get("local_chunks", []),
        course_code=state["course_code"]
    )
    
    citations = [
        {
            "source": c.get("metadata", {}).get("source", "Course Document"),
            "page": c.get("metadata", {}).get("page", "?"),
            "score": round(c.get("score", 0.0), 4),
            "excerpt": c.get("text", "")[:400] + ("..." if len(c.get("text", "")) > 400 else "")
        }
        for c in routed_context
    ]
    
    return {
        "response_type": "TECHNICAL",
        "crag_decision": decision,
        "final_context": routed_context,
        "citations": citations
    }


def route_by_intent(state: GraphState) -> str:
    return "handle_conversational" if state["intent"] == "CONVERSATIONAL" else "execute_retrieval"

workflow = StateGraph(GraphState)

workflow.add_node("condenser", condense_node)
workflow.add_node("conversational_handler", conversational_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("crag", crag_eval_node)

workflow.add_edge(START, "condenser")
workflow.add_conditional_edges(
    "condenser",
    route_by_intent,
    {
        "handle_conversational": "conversational_handler",
        "execute_retrieval": "retriever"
    }
)
workflow.add_edge("conversational_handler", END)
workflow.add_edge("retriever", "crag")
workflow.add_edge("crag", END)

kognit_graph = workflow.compile()