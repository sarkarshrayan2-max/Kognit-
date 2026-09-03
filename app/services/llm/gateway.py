import os
import re
from typing import Any, Dict, Iterator, List

from groq import Groq


class LLMGateway:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        self.client = Groq(api_key=api_key)

        self.model_name = os.getenv(
            "GROQ_MODEL",
            "qwen/qwen3.6-27b",
        )

    # ============================================================
    # CONTEXT FORMATTING
    # ============================================================

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

            text = chunk.get(
                "text",
                "",
            ).strip()

            if not text:
                continue

            # ----------------------------------------------------
            # Source label
            # ----------------------------------------------------

            if source_type == "web":
                label = "EXTERNAL WEB SOURCE"
            else:
                label = "COURSE DOCUMENT"

            formatted_chunks.append(
                f"""
REFERENCE {index}
TYPE: {label}
COURSE: {course_code}
SOURCE: {source}
PAGE: {page}

CONTENT:
{text}
""".strip()
            )

        if not formatted_chunks:
            return "No usable reference material was retrieved."

        return "\n\n".join(
            formatted_chunks
        )

    # ============================================================
    # SYSTEM PROMPT
    # ============================================================

    @staticmethod
    def _build_system_prompt(
        crag_decision: str,
    ) -> str:

        decision = (
            crag_decision or "UNKNOWN"
        ).upper()

        # --------------------------------------------------------
        # CORRECT
        # --------------------------------------------------------

        if decision == "CORRECT":

            source_instruction = """
The retrieved course material is considered sufficiently relevant.

Use the COURSE DOCUMENT material as the primary and authoritative
source for your answer.

Do not introduce information that contradicts the course material.

If the answer cannot be established from the course material,
say so.
""".strip()

        # --------------------------------------------------------
        # AMBIGUOUS
        # --------------------------------------------------------

        elif decision == "AMBIGUOUS":

            source_instruction = """
The retrieved course material is partially relevant, and external
web material has been included as supplementary information.

PRIORITY ORDER:

1. COURSE DOCUMENT
2. EXTERNAL WEB SOURCE

Use the course material as the primary basis of the answer.

Use external sources only when they help clarify, supplement,
or fill a gap in the course material.

Do not present external information as if it came from the
course documents.

If an external source materially contributes to the answer,
make that clear.
""".strip()

        # --------------------------------------------------------
        # INCORRECT
        # --------------------------------------------------------

        elif decision == "INCORRECT":

            source_instruction = """
The local course material was not sufficiently relevant, so
external web sources were retrieved.

Answer using the available external sources when they are relevant.

Do NOT pretend that external information came from the student's
course documents.

Clearly indicate when the answer is based on external sources.

If the retrieved external sources are insufficient, say that
reliable information could not be established.
""".strip()

        # --------------------------------------------------------
        # UNKNOWN
        # --------------------------------------------------------

        else:

            source_instruction = """
Use the retrieved references carefully.

Prefer COURSE DOCUMENT sources over EXTERNAL WEB SOURCES.

Do not invent information.

Do not claim that information came from a source when it did not.
""".strip()

        # --------------------------------------------------------
        # FINAL SYSTEM PROMPT
        # --------------------------------------------------------

        return f"""
You are KOGNIT, an academic assistant for engineering students.

Your job is to answer questions accurately using the provided
reference material.

{source_instruction}

GENERAL RULES:

1. Answer the user's actual question directly.

2. Ground factual claims in the provided reference material.

3. Do not fabricate facts, citations, page numbers, or sources.

4. Do not mention internal system details such as:
   - CRAG
   - retrieval scores
   - embeddings
   - vector databases
   - rerankers
   - LangGraph

   unless the user explicitly asks about the system.

5. If the references do not contain enough information, say so
   instead of hallucinating an answer.

6. Explain technical concepts at an appropriate engineering-student
   level.

7. Use equations when they improve understanding.

8. For comparisons, use a concise table when appropriate.

9. Do not reproduce large portions of the source material verbatim.

10. Keep the answer focused and avoid unnecessary repetition.

11. If a source contains conflicting information, explicitly mention
    the conflict rather than silently choosing one.

12. Never treat a web source as a course document.

13. Never output internal reasoning or hidden analysis.

14. Never output a scratchpad.

15. Never output phrases such as:
    "Here's a thinking process:"
    "Let's think step by step"
    "Analysis:"
    "Reasoning:"
    "I need to analyze"
    "First, I will analyze"
    "Output Generation:"
    "Final Response:"

16. Do not explain how you selected or ranked the sources.

17. Do not describe your internal reasoning process.

18. Output ONLY the final student-facing answer.

19. Start the response immediately with the answer.

The CRAG routing decision for this response is:

{decision}
""".strip()

    # ============================================================
    # MESSAGE BUILDER
    # ============================================================

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

        # --------------------------------------------------------
        # CHAT HISTORY
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # CURRENT USER MESSAGE
        # --------------------------------------------------------

        user_message = f"""
REFERENCE MATERIAL
==================

{context}


USER QUESTION
=============

{query}


INSTRUCTIONS
============

Answer the user's question using the reference material and the
source-priority rules provided by the system instructions.

Return only the final answer for the student.

Do not output internal reasoning, scratchpad content, analysis,
or source-selection reasoning.
""".strip()

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return messages

    # ============================================================
    # MODEL OUTPUT CLEANER
    # ============================================================

    @staticmethod
    def _clean_model_output(
        text: str,
    ) -> str:

        if not text:
            return ""

        cleaned = text.strip()

        # --------------------------------------------------------
        # Remove <think>...</think>
        # --------------------------------------------------------

        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # --------------------------------------------------------
        # Remove unfinished <think> blocks
        # --------------------------------------------------------

        cleaned = re.sub(
            r"<think>.*$",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # --------------------------------------------------------
        # Remove closing think tag if it appears alone
        # --------------------------------------------------------

        cleaned = re.sub(
            r"</think>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # --------------------------------------------------------
        # Remove common reasoning prefixes
        # --------------------------------------------------------

        reasoning_prefixes = [
            r"Here's a thinking process:",
            r"Here is a thinking process:",
            r"Here\'s my thinking process:",
            r"Here is my thinking process:",
            r"Let's think step by step:",
            r"Let\'s think step by step:",
            r"Thinking process:",
            r"Reasoning:",
            r"Analysis:",
            r"Internal reasoning:",
            r"Scratchpad:",
            r"Output Generation:",
            r"Final Response:",
        ]

        for prefix in reasoning_prefixes:

            cleaned = re.sub(
                prefix,
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        # --------------------------------------------------------
        # If the model explicitly generated a thinking section,
        # discard everything before the actual answer.
        #
        # KOGNIT's prompt asks the model to begin directly with
        # the final answer, so these are safe recovery markers.
        # --------------------------------------------------------

        answer_markers = [
            "**Intuitive Concept**",
            "**Formal Definition**",
            "**Comparative Breakdown**",
            "**Formula & Syntax Breakdown**",
            "**Answer**",
            "Based on your course material",
        ]

        positions = []

        for marker in answer_markers:

            position = cleaned.find(
                marker
            )

            if position != -1:
                positions.append(
                    position
                )

        if positions:

            first_answer_position = min(
                positions
            )

            # Only discard preceding content when it looks like
            # reasoning contamination.
            preceding = cleaned[
                :first_answer_position
            ]

            reasoning_indicators = [
                "analyze",
                "analysis",
                "reasoning",
                "thinking",
                "identify",
                "draft",
                "check",
                "output generation",
                "final response",
                "reference material",
                "source selection",
            ]

            if any(
                indicator in preceding.lower()
                for indicator in reasoning_indicators
            ):

                cleaned = cleaned[
                    first_answer_position:
                ]

        # --------------------------------------------------------
        # Remove excessive blank lines
        # --------------------------------------------------------

        cleaned = re.sub(
            r"\n{3,}",
            "\n\n",
            cleaned,
        )

        return cleaned.strip()

    # ============================================================
    # STREAM ANSWER
    # ============================================================

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

        # --------------------------------------------------------
        # GROQ STREAM
        # --------------------------------------------------------

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
            stream=True,
        )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT clean each individual provider chunk.
        #
        # Example:
        #
        # chunk 1 = "Here's a think"
        # chunk 2 = "ing process:"
        #
        # Cleaning them independently will fail.
        #
        # Therefore we first collect the complete model output,
        # clean it, and then yield the clean answer.
        # --------------------------------------------------------

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

            if not content:
                continue

            accumulated_output.append(
                content
            )

        # --------------------------------------------------------
        # COMPLETE MODEL OUTPUT
        # --------------------------------------------------------

        raw_output = "".join(
            accumulated_output
        )

        # --------------------------------------------------------
        # SANITIZE
        # --------------------------------------------------------

        cleaned_output = (
            self._clean_model_output(
                raw_output
            )
        )

        if not cleaned_output:
            return

        # --------------------------------------------------------
        # STREAM CLEANED ANSWER
        #
        # This preserves the frontend's existing streaming/SSE
        # behavior while ensuring that hidden reasoning cannot
        # leak into the response.
        # --------------------------------------------------------

        chunk_size = 80

        for start in range(
            0,
            len(cleaned_output),
            chunk_size,
        ):

            yield cleaned_output[
                start:start + chunk_size
            ]