import os
import re
from typing import Any, Dict, Iterator, List

from groq import Groq


class LLMGateway:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv(
            "GROQ_MODEL",
            "qwen/qwen3.6-27b",
        )
        self.client = Groq(api_key=api_key) if api_key else None

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

        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {})

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
                reference += f"URL: {url}\n"

            reference += f"\nCONTENT:\n{text}"

            formatted_chunks.append(reference)

        if not formatted_chunks:
            return "No usable reference material was retrieved."

        return "\n\n".join(formatted_chunks)

    @staticmethod
    def _build_system_prompt(
        crag_decision: str,
    ) -> str:
        decision = (crag_decision or "UNKNOWN").upper()

        if decision == "CORRECT":
            source_instruction = """
The retrieved course material is sufficiently relevant.

Use the COURSE DOCUMENT material as the primary and authoritative
source for the answer.

Answer only from information supported by the course material.
""".strip()

        elif decision == "AMBIGUOUS":
            source_instruction = """
The retrieved material contains both course and external sources.

Use the COURSE DOCUMENT material as the primary source.

Use EXTERNAL WEB SOURCES only to clarify or supplement information
when necessary.

Never present web information as if it came from the course material.
""".strip()

        elif decision == "INCORRECT":
            source_instruction = """
The local course material was not sufficiently relevant.

The available reference material therefore comes from EXTERNAL WEB
SOURCES.

Answer the user's question using the external references when they
contain relevant information.

Do not claim that external information came from the course documents.

Do not say that the course material contains the answer when it does not.

If the external references are insufficient, clearly state that the
available sources do not provide enough information.
""".strip()

        else:
            source_instruction = """
Use the retrieved references carefully.

Prefer COURSE DOCUMENT sources when relevant.

Use EXTERNAL WEB SOURCES when they are the relevant available sources.

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
5. Do not mention internal system details such as CRAG, retrieval,
embeddings, vector databases, rerankers, or LangGraph unless the
student explicitly asks about them.
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
19. Never claim that information came from the course document when
the source is external.

The CRAG routing decision is:

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
                role = message.get("role")

                if role not in {"user", "assistant"}:
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

INSTRUCTIONS
============

Answer the user's question using the reference material.

The current source routing decision is: {crag_decision.upper()}

If the routing decision is INCORRECT, the external web references are
the relevant sources for this question.

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
    def _clean_model_output(text: str) -> str:
        if not text:
            return ""

        cleaned = text.strip()

        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        cleaned = re.sub(
            r"<think>.*$",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        cleaned = re.sub(
            r"</think>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"^(Here's a thinking process:|Here is a thinking process:|"
            r"Here's my thinking process:|Here is my thinking process:|"
            r"Let's think step by step:?|Thinking process:|Reasoning:|"
            r"Analysis:|Internal reasoning:|Scratchpad:|"
            r"Output Generation:|Final Response:)\s*",
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
            temperature=0.2,
            max_tokens=700,
            reasoning_format="hidden",
            stream=True,
        )

    def stream_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        crag_decision: str = "UNKNOWN",
    ) -> Iterator[str]:
        messages = self._build_messages(
            query=query,
            retrieved_chunks=retrieved_chunks,
            history=history,
            crag_decision=crag_decision,
        )

        stream = self._create_completion(messages)

        accumulated_output = []

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            content = getattr(
                delta,
                "content",
                None,
            )

            if content:
                accumulated_output.append(content)
                continue

            reasoning = getattr(
                delta,
                "reasoning",
                None,
            )

            if reasoning:
                accumulated_output.append(reasoning)

        raw_output = "".join(accumulated_output)

        cleaned_output = self._clean_model_output(
            raw_output
        )

        if not cleaned_output:
            fallback_response = self._fallback_completion(
                messages
            )

            cleaned_output = self._clean_model_output(
                fallback_response
            )

        if not cleaned_output:
            return

        chunk_size = 80

        for start in range(
            0,
            len(cleaned_output),
            chunk_size,
        ):
            yield cleaned_output[
                start:start + chunk_size
            ]

    def _fallback_completion(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        client = self._require_client()

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=700,
            reasoning_format="hidden",
            stream=False,
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

        reasoning = getattr(
            message,
            "reasoning",
            None,
        )

        if reasoning:
            return reasoning

        return ""