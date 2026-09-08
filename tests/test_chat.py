import json

import app.services.retrieval.fusion as fusion


# =========================================================
# LIGHTWEIGHT RETRIEVER MOCK
# =========================================================
#
# chat.py imports the real graph workflow.
# The workflow normally creates HybridRetriever(), which
# loads SentenceTransformer/CrossEncoder models.
#
# Replace only the constructor dependency before importing
# chat.py. This keeps the real app.graph.workflow module
# intact and avoids sys.modules pollution.
#


class FakeHybridRetriever:
    def search(
        self,
        query,
        course_code,
        top_k=3,
        candidate_limit=10,
        rrf_k=60,
    ):
        return []


fusion.HybridRetriever = FakeHybridRetriever


# =========================================================
# IMPORT REAL chat.py
# =========================================================

import app.api.chat as chat


# =========================================================
# TEST 1 — SSE EVENT FORMAT
# =========================================================

def test_sse_event_format():
    payload = {
        "type": "token",
        "content": "Hello",
    }

    result = chat.sse_event(payload)

    assert result.startswith("data: ")
    assert result.endswith("\n\n")

    json_payload = result[len("data: "):].strip()

    decoded = json.loads(json_payload)

    assert decoded == payload


# =========================================================
# TEST 2 — SSE METADATA EVENT
# =========================================================

def test_sse_metadata_event():
    payload = {
        "type": "metadata",
        "crag_decision": "CORRECT",
        "citations": [
            {
                "source": "test.pdf",
                "page": 5,
                "score": 0.9,
            }
        ],
        "model_used": "test-model",
        "standalone_query": "What is normalization?",
        "response_type": "TECHNICAL",
    }

    result = chat.sse_event(payload)

    assert result.startswith("data: ")
    assert result.endswith("\n\n")

    decoded = json.loads(
        result[len("data: "):].strip()
    )

    assert decoded["type"] == "metadata"
    assert decoded["crag_decision"] == "CORRECT"
    assert decoded["response_type"] == "TECHNICAL"
    assert decoded["model_used"] == "test-model"

    assert len(decoded["citations"]) == 1


# =========================================================
# TEST 3 — SSE ERROR EVENT
# =========================================================

def test_sse_error_event():
    payload = {
        "type": "error",
        "content": (
            "An internal error occurred "
            "while processing your request."
        ),
    }

    result = chat.sse_event(payload)

    assert result.startswith("data: ")
    assert result.endswith("\n\n")

    decoded = json.loads(
        result[len("data: "):].strip()
    )

    assert decoded["type"] == "error"

    assert (
        decoded["content"]
        == "An internal error occurred "
           "while processing your request."
    )


# =========================================================
# TEST 4 — SSE PRESERVES UNICODE
# =========================================================

def test_sse_event_preserves_unicode():
    payload = {
        "type": "token",
        "content": "Normalization → कमी Redundancy",
    }

    result = chat.sse_event(payload)

    decoded = json.loads(
        result[len("data: "):].strip()
    )

    assert decoded["content"] == payload["content"]


# =========================================================
# TEST 5 — SSE DONE EVENT
# =========================================================

def test_sse_done_event():
    payload = {
        "type": "done",
    }

    result = chat.sse_event(payload)

    decoded = json.loads(
        result[len("data: "):].strip()
    )

    assert decoded == {
        "type": "done",
    }