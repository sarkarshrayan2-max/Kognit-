import logging
import os
import re
from typing import Any, Dict, Iterator, List

from groq import Groq

logger = logging.getLogger("kognit.llm")


class LLMGateway:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        self.model_name = os.getenv(
            "GROQ_MODEL",
            "qwen/qwen3.6-27b",
        )

        self.client = (
            Groq(api_key=api_key)
            if api_key
            else None
        )

    def _require_client(self) -> Groq:
        if self.client is None:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. "
                "Set GROQ_API_KEY in your .env file."
            )

        return self.client

    @staticmethod
    def _format_context(
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:

        if not retrieved_chunks:
            return "No reference material was retrieved."

        formatted_chunks = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            metadata = chunk.get(
                "metadata",
                {},
            )

            source_type = metadata.get(
                "source_type",
                "course",
            )

            source = metadata.get(
                "source",
                "Unknown source",
            )

            page = metadata.get(
                "page",
                "?",
            )

            course_code = metadata.get(
                "course_code",
                "Unknown",
            )

            url = metadata.get(
                "url",
                "",
            )

            text = chunk.get(
                "text",
                "",
            ).strip()

            if not text:
                continue

            label = (
                "EXTERNAL WEB SOURCE"
                if source_type == "web"
                else "COURSE DOCUMENT"
            )

            reference = (
                f"REFERENCE {index}\n"
                f"TYPE: {label}\n"
                f"COURSE: {course_code}\n"
                f"SOURCE: {source}\n"
                f"PAGE: {page}\n"
            )

            if url:
                reference += (
                    f"URL: {url}\n"
                )

            reference += (
                f"\nCONTENT:\n{text}"
            )

            formatted_chunks.append(
                reference
            )

        if not formatted_chunks:
            return (
                "No usable reference material "
                "was retrieved."
            )

        return "\n\n".join(
            formatted_chunks
        )

    @staticmethod
    def _build_system_prompt(
        crag_decision: str,
    ) -> str:

        decision = (
            crag_decision or "UNKNOWN"
        ).upper().strip()

        if decision == "CORRECT":

            source_instruction = """
The retrieved course material is sufficiently relevant.

Use the COURSE DOCUMENT material as the primary and authoritative
source for the answer.

Answer only from information supported by the course material.
""".strip()

        elif decision == "WEB_FALLBACK":

            source_instruction = """
The question is relevant to the selected course, but the course
documents do not contain sufficient information.

Use the provided EXTERNAL WEB SOURCES to answer the question.

The external sources were selected using the active course scope.

Never present external information as if it came from a course
document.

If course-document information and external information differ,
clearly distinguish them.
""".strip()

        elif decision == "INSUFFICIENT":

            source_instruction = """
The question is relevant to the selected course, but the available
course material is insufficient.

Use only the provided COURSE DOCUMENT material.

Do not invent missing information.

If the available material cannot answer the question, explicitly state
that the available course material is insufficient.
""".strip()

        elif decision == "NOT_FOUND":

            source_instruction = """
No sufficiently relevant reference material was found.

Do not invent an answer.

Clearly state that sufficient information was not found.
""".strip()

        elif decision == "OUT_OF_SCOPE":

            source_instruction = """
The question is outside the scope of the selected course.

Do not answer the technical question.

The application should normally handle this decision before reaching
the LLM generation stage.
""".strip()

        elif decision == "OFF_TOPIC":

            source_instruction = """
The student's message is not a technical or academic question at all
(e.g. it concerns literature, entertainment, general trivia, or another
subject unrelated to engineering coursework).

Do not answer the underlying non-technical topic.

Briefly and politely note that you can only help with technical and
academic questions for the student's courses, and invite them to ask
one.

The application should normally handle this decision before reaching
the LLM generation stage.
""".strip()

        else:

            source_instruction = """
Use the provided reference material carefully.

Prefer COURSE DOCUMENT material when it is relevant.

Use EXTERNAL WEB SOURCES only when the routing decision explicitly
allows external sources.

Do not invent information or source attribution.
""".strip()

        return f"""
You are KOGNIT, an academic assistant for engineering students.

Your task is to answer the student's question accurately using the
provided reference material.

{source_instruction}

GENERAL RULES:

1. Answer the user's actual question directly.
2. Ground factual claims in the provided references.
3. Do not fabricate facts, citations, page numbers, URLs, or sources.
4. Never treat an EXTERNAL WEB SOURCE as a COURSE DOCUMENT.
5. Do not mention CRAG, retrieval, embeddings, vector databases,
rerankers, LangGraph, or internal system details unless the student
explicitly asks about them.
6. If the references do not contain enough information, say so.
7. Explain technical concepts at an engineering-student level.
8. Use examples when they improve understanding.
9. Use equations when appropriate.
10. Use concise tables for comparisons when useful.
11. Do not reproduce large portions of source material verbatim.
12. Keep the answer focused.
13. Do not output internal reasoning.
14. Do not output a scratchpad.
15. Do not output analysis or source-selection reasoning.
16. Output only the final student-facing answer.
17. Start immediately with the answer.
18. Do not explain how sources were selected or ranked.
19. Never claim that information came from a course document when the
source is external.
20. Never use external sources when the routing decision does not allow
them.

The current routing decision is:

{decision}
""".strip()

    def _build_messages(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        crag_decision: str,
    ) -> List[Dict[str, str]]:

        system_prompt = self._build_system_prompt(
            crag_decision=crag_decision,
        )

        context = self._format_context(
            retrieved_chunks=retrieved_chunks,
        )

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if history:

            for message in history[-4:]:

                role = message.get(
                    "role"
                )

                if role not in {
                    "user",
                    "assistant",
                }:
                    continue

                content = message.get(
                    "content",
                    "",
                ).strip()

                if not content:
                    continue

                messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        user_message = f"""
REFERENCE MATERIAL
==================

{context}

USER QUESTION
=============

{query}

ROUTING DECISION
================

{crag_decision.upper()}

SOURCE RULES
============

CORRECT:
Use the provided course documents.

WEB_FALLBACK:
Use the provided external web sources and any useful course material.
Do not claim external information came from the course documents.

INSUFFICIENT:
Use only the provided course material and explicitly acknowledge
missing information when necessary.

NOT_FOUND:
Do not invent an answer.

OUT_OF_SCOPE:
Do not answer the technical question. The application should normally
block generation for this decision.

OFF_TOPIC:
Do not answer the non-technical topic. The application should normally
block generation for this decision.

Return only the final answer for the student.

Do not output internal reasoning, analysis, scratchpad content,
source-selection reasoning, or system details.
""".strip()

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return messages

    @staticmethod
    def _clean_model_output(
        text: str,
    ) -> str:

        if not text:
            return ""

        cleaned = text.strip()

        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned,
            flags=(
                re.DOTALL
                | re.IGNORECASE
            ),
        )

        cleaned = re.sub(
            r"<think>.*$",
            "",
            cleaned,
            flags=(
                re.DOTALL
                | re.IGNORECASE
            ),
        )

        cleaned = re.sub(
            r"</think>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"^(Here's a thinking process:|"
            r"Here is a thinking process:|"
            r"Here's my thinking process:|"
            r"Here is my thinking process:|"
            r"Let's think step by step:?|"
            r"Thinking process:|"
            r"Reasoning:|"
            r"Analysis:|"
            r"Internal reasoning:|"
            r"Scratchpad:|"
            r"Output Generation:|"
            r"Final Response:)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\n{3,}",
            "\n\n",
            cleaned,
        )

        return cleaned.strip()

    def _create_completion(
        self,
        messages: List[Dict[str, str]],
    ):

        client = self._require_client()

        return client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=900,
            reasoning_effort="none",
            stream=True,
        )

    def stream_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        crag_decision: str = "UNKNOWN",
    ) -> Iterator[str]:

        decision = crag_decision.upper()

        if decision == "OUT_OF_SCOPE":

            yield (
                "This question is outside the scope of the selected "
                "course. Please select the appropriate course to ask "
                "this question."
            )

            return

        if decision == "OFF_TOPIC":

            yield (
                "That's outside what I can help with here — I'm built "
                "to assist with technical and academic questions for "
                "your engineering courses. Feel free to ask me "
                "something related to your coursework instead."
            )

            return

        messages = self._build_messages(
            query=query,
            retrieved_chunks=retrieved_chunks,
            history=history,
            crag_decision=crag_decision,
        )

        received_content = False

        try:

            stream = self._create_completion(
                messages
            )

            for chunk in stream:

                if not chunk.choices:
                    continue

                delta = (
                    chunk.choices[0].delta
                )

                content = getattr(
                    delta,
                    "content",
                    None,
                )

                if not content:
                    continue

                received_content = True

                yield content

        except Exception:

            logger.exception(
                "Groq streaming generation failed"
            )

            if not received_content:

                try:

                    fallback_response = (
                        self._fallback_completion(
                            messages
                        )
                    )

                    cleaned = (
                        self._clean_model_output(
                            fallback_response
                        )
                    )

                    if cleaned:
                        yield cleaned
                        return

                except Exception:

                    logger.exception(
                        "Groq fallback generation failed"
                    )

        if not received_content:

            yield (
                "I couldn't generate an answer "
                "right now. Please try the "
                "question again."
            )

    def _fallback_completion(
        self,
        messages: List[Dict[str, str]],
    ) -> str:

        client = self._require_client()

        response = (
            client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=900,
                reasoning_effort="none",
                stream=False,
            )
        )

        if not response.choices:
            return ""

        message = response.choices[0].message

        content = getattr(
            message,
            "content",
            None,
        )

        if content:
            return content

        return ""