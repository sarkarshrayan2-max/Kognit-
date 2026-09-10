from typing import Any, Dict, Iterator, List, Optional

from app.services.llm.prompts import build_messages


class AnswerGenerator:
    def __init__(self, gateway):
        self.gateway = gateway

    def build_messages(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        crag_decision: str,
        long_term_memories: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> List[Dict[str, str]]:
        return build_messages(
            query=query,
            retrieved_chunks=retrieved_chunks,
            history=history,
            crag_decision=crag_decision,
            long_term_memories=long_term_memories or [],
        )

    def stream(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        crag_decision: str = "UNKNOWN",
        long_term_memories: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> Iterator[str]:
        decision = (crag_decision or "UNKNOWN").upper().strip()

        if decision == "OUT_OF_SCOPE":
            yield (
                "This question is outside the scope "
                "of the selected course. Please select "
                "the appropriate course to ask this question."
            )
            return

        if decision == "OFF_TOPIC":
            yield (
                "That's outside what I can help with here — "
                "I'm built to assist with technical and academic "
                "questions for your engineering courses. "
                "Feel free to ask me something related to "
                "your coursework instead."
            )
            return

        messages = self.build_messages(
            query=query,
            retrieved_chunks=retrieved_chunks,
            history=history,
            crag_decision=decision,
            long_term_memories=long_term_memories,
        )

        yield from self.gateway.stream_completion(messages)

    def complete(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        crag_decision: str = "UNKNOWN",
        long_term_memories: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> str:
        decision = (crag_decision or "UNKNOWN").upper().strip()

        if decision == "OUT_OF_SCOPE":
            return (
                "This question is outside the scope "
                "of the selected course. Please select "
                "the appropriate course to ask this question."
            )

        if decision == "OFF_TOPIC":
            return (
                "That's outside what I can help with here — "
                "I'm built to assist with technical and academic "
                "questions for your engineering courses. "
                "Feel free to ask me something related to "
                "your coursework instead."
            )

        messages = self.build_messages(
            query=query,
            retrieved_chunks=retrieved_chunks,
            history=history,
            crag_decision=decision,
            long_term_memories=long_term_memories,
        )

        return self.gateway.create_completion(messages)