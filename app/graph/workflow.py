import hashlib
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.services.llm.gateway import LLMGateway
from app.services.rag.condenser import QueryCondenser
from app.services.rag.crag import CRAGEvaluator
from app.services.retrieval.fusion import HybridRetriever


logger = logging.getLogger(
    "kognit.workflow"
)



retriever = HybridRetriever()

crag_evaluator = CRAGEvaluator()

condenser = QueryCondenser()

llm_gateway = LLMGateway()



class GraphState(TypedDict, total=False):

    query: str

    course_code: str

    history: List[
        Dict[str, str]
    ]

    top_k: int

    intent: str

    standalone_query: str

    response_type: str

    local_chunks: List[
        Dict[str, Any]
    ]

    final_context: List[
        Dict[str, Any]
    ]

    crag_decision: str

    citations: List[
        Dict[str, Any]
    ]

    answer: Optional[str]



def condense_node(
    state: GraphState,
) -> Dict[str, Any]:

    intent, standalone_query = (
        condenser.analyze(
            query=state["query"],
            history=state.get(
                "history",
                [],
            ),
            course_code=
                state["course_code"],
        )
    )

    if not standalone_query:
        standalone_query = (
            state["query"]
        )

    logger.info(
        "Intent=%s | query=%s | standalone=%s",
        intent,
        state["query"],
        standalone_query,
    )

    return {
        "intent": intent,
        "standalone_query":
            standalone_query,
    }



def conversational_node(
    state: GraphState,
) -> Dict[str, Any]:

    answer = (
        "Understood! Let me know if "
        "you want to explore more "
        "examples or dive into another "
        "topic."
    )

    return {
        "response_type":
            "CONVERSATIONAL",

        "answer": answer,

        "crag_decision":
            "CONVERSATIONAL",

        "final_context": [],

        "citations": [],
    }



def retrieve_node(
    state: GraphState,
) -> Dict[str, Any]:

    top_k = state.get(
        "top_k",
        3,
    )

    chunks = retriever.search(
        query=
            state["standalone_query"],

        course_code=
            state["course_code"],

        top_k=top_k,
    )

    logger.info(
        "Retrieved %d local chunks "
        "for course=%s",
        len(chunks),
        state["course_code"],
    )

    return {
        "local_chunks": chunks,
    }




def deduplicate_chunks(
    chunks: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:

    unique = {}

    for chunk in chunks:

        metadata = chunk.get(
            "metadata",
            {},
        )

        document_id = metadata.get(
            "document_id",
            "",
        )

        source = metadata.get(
            "source",
            "",
        )

        page = metadata.get(
            "page",
            "",
        )

        chunk_index = metadata.get(
            "chunk_index",
            "",
        )

        text = " ".join(
            chunk.get(
                "text",
                "",
            ).split()
        ).strip()

        if (
            document_id
            or chunk_index
        ):

            key = (
                "metadata",
                document_id,
                source,
                page,
                chunk_index,
            )

        else:

            text_hash = hashlib.sha256(
                text.lower().encode(
                    "utf-8"
                )
            ).hexdigest()

            key = (
                "text",
                source,
                page,
                text_hash,
            )

        existing = unique.get(
            key
        )

        if existing is None:

            unique[key] = chunk

            continue

        existing_score = float(
            existing.get(
                "score",
                0.0,
            )
        )

        current_score = float(
            chunk.get(
                "score",
                0.0,
            )
        )

        if current_score > existing_score:

            unique[key] = chunk

    return list(
        unique.values()
    )




def crag_eval_node(
    state: GraphState,
) -> Dict[str, Any]:

    decision, routed_context = (
        crag_evaluator
        .evaluate_and_route(
            query=
                state[
                    "standalone_query"
                ],

            local_chunks=
                state.get(
                    "local_chunks",
                    [],
                ),

            course_code=
                state["course_code"],
        )
    )

    routed_context = (
        deduplicate_chunks(
            routed_context
        )
    )

    citations = []

    for chunk in routed_context:

        metadata = chunk.get(
            "metadata",
            {},
        )

        source_type = metadata.get(
            "source_type",
            "course",
        )

        page = metadata.get(
            "page",
            "?",
        )

        document_id = metadata.get(
            "document_id",
            "",
        )

        source = metadata.get(
            "source",
            "Course Document",
        )

    

        if source_type == "course":

            url = (
                f"/documents/files/"
                f"{document_id}"
                f"#page={page}"
                if document_id
                else ""
            )

        elif source_type == "web":

            url = metadata.get(
                "url",
                "",
            )

        else:

            url = ""

        text = chunk.get(
            "text",
            "",
        )

        citations.append(
            {
                "source": source,

                "page": page,

                "score": round(
                    float(
                        chunk.get(
                            "score",
                            0.0,
                        )
                    ),
                    4,
                ),

                "excerpt": (
                    text[:400]
                    + (
                        "..."
                        if len(text) > 400
                        else ""
                    )
                ),

                "source_type":
                    source_type,

                "document_id":
                    document_id,

                "url": url,
            }
        )

    return {
        "response_type":
            "TECHNICAL",

        "crag_decision":
            decision,

        "final_context":
            routed_context,

        "citations":
            citations,
    }



def generation_node(
    state: GraphState,
) -> Dict[str, Any]:

    final_context = state.get(
        "final_context",
        [],
    )

    if not final_context:

        answer = (
            "No sufficient material was "
            "found in the course documents "
            "or permitted external sources."
        )

        writer = get_stream_writer()

        writer(
            {
                "type": "token",
                "content": answer,
            }
        )

        return {
            "answer": answer
        }


    writer = get_stream_writer()

    answer_parts: List[str] = []

    try:

        for token in (
            llm_gateway.stream_answer(
                query=state["query"],

                retrieved_chunks=
                    final_context,

                history=state.get(
                    "history",
                    [],
                ),

                crag_decision=
                    state.get(
                        "crag_decision",
                        "UNKNOWN",
                    ),
            )
        ):

            if not token:
                continue

            answer_parts.append(
                token
            )


            writer(
                {
                    "type": "token",
                    "content": token,
                }
            )

    except Exception:

        logger.exception(
            "LLM generation failed"
        )

        error_message = (
            "I encountered an error "
            "while generating the answer."
        )

        writer(
            {
                "type": "token",
                "content": error_message,
            }
        )

        return {
            "answer":
                error_message
        }

    answer = "".join(
        answer_parts
    )

    return {
        "answer": answer,
    }



def route_by_intent(
    state: GraphState,
) -> str:

    if (
        state.get("intent")
        == "CONVERSATIONAL"
    ):

        return (
            "handle_conversational"
        )

    return "execute_retrieval"




workflow = StateGraph(
    GraphState
)


workflow.add_node(
    "condenser",
    condense_node,
)

workflow.add_node(
    "conversational_handler",
    conversational_node,
)

workflow.add_node(
    "retriever",
    retrieve_node,
)

workflow.add_node(
    "crag",
    crag_eval_node,
)

workflow.add_node(
    "generator",
    generation_node,
)


workflow.add_edge(
    START,
    "condenser",
)


workflow.add_conditional_edges(
    "condenser",
    route_by_intent,
    {
        "handle_conversational":
            "conversational_handler",

        "execute_retrieval":
            "retriever",
    },
)


workflow.add_edge(
    "conversational_handler",
    END,
)


workflow.add_edge(
    "retriever",
    "crag",
)


workflow.add_edge(
    "crag",
    "generator",
)


workflow.add_edge(
    "generator",
    END,
)

kognit_graph = workflow.compile()