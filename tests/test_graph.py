from unittest.mock import MagicMock

import app.graph.workflow as workflow

from app.graph.workflow import (
    crag_eval_node,
    retrieve_node,
)


# =========================================================
# FAKE RETRIEVER
# =========================================================

class FakeHybridRetriever:
    def search(
        self,
        query,
        course_code,
        top_k=3,
        candidate_limit=10,
        rrf_k=60,
    ):
        return [
            {
                "score": 0.90,
                "text": (
                    "Normalization organizes database tables "
                    "to reduce redundancy."
                ),
                "metadata": {
                    "source": "test.pdf",
                    "page": 1,
                    "course_code": course_code,
                },
            }
        ]


# =========================================================
# TEST 1 — RETRIEVAL NODE
# =========================================================

def test_retrieve_node_returns_chunks(monkeypatch):
    fake_retriever = FakeHybridRetriever()

    monkeypatch.setattr(
        workflow,
        "retriever",
        fake_retriever,
    )

    state = {
        "query": "What is normalization?",
        "course_code": "DBMS",
        "standalone_query": "What is normalization?",
        "top_k": 3,
    }

    result = retrieve_node(state)

    assert "local_chunks" in result
    assert isinstance(result["local_chunks"], list)

    assert len(result["local_chunks"]) == 1

    chunk = result["local_chunks"][0]

    assert chunk["text"]
    assert chunk["score"] == 0.90

    assert chunk["metadata"]["course_code"] == "DBMS"
    assert chunk["metadata"]["source"] == "test.pdf"
    assert chunk["metadata"]["page"] == 1


# =========================================================
# TEST 2 — CRAG NODE
# =========================================================

def test_crag_node_builds_citations(monkeypatch):
    fake_evaluator = MagicMock()

    fake_evaluator.evaluate_and_route.return_value = (
        "CORRECT",
        [
            {
                "score": 0.90,
                "text": (
                    "Normalization reduces redundancy "
                    "in databases."
                ),
                "metadata": {
                    "source": "test.pdf",
                    "page": 5,
                },
            }
        ],
    )

    monkeypatch.setattr(
        workflow,
        "crag_evaluator",
        fake_evaluator,
    )

    state = {
        "standalone_query": "What is normalization?",
        "course_code": "DBMS",
        "local_chunks": [
            {
                "score": 0.90,
                "text": (
                    "Normalization reduces redundancy "
                    "in databases."
                ),
                "metadata": {
                    "source": "test.pdf",
                    "page": 5,
                },
            }
        ],
    }

    result = crag_eval_node(state)

    assert result["crag_decision"] == "CORRECT"

    assert "final_context" in result
    assert len(result["final_context"]) == 1

    assert "citations" in result
    assert len(result["citations"]) == 1

    citation = result["citations"][0]

    assert citation["source"] == "test.pdf"
    assert citation["page"] == 5
    assert citation["score"] == 0.90