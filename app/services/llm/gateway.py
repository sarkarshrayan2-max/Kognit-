import logging
import os
import re
from typing import Dict, Iterator, List

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

        self._generator = None

    def _require_client(self) -> Groq:
        if self.client is None:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. "
                "Set GROQ_API_KEY in your .env file."
            )

        return self.client

    def _get_generator(self):
        if self._generator is None:
            from app.services.llm.generation import AnswerGenerator

            self._generator = AnswerGenerator(self)

        return self._generator

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

    def stream_completion(
        self,
        messages: List[Dict[str, str]],
    ) -> Iterator[str]:
        received_content = False

        try:
            stream = self._create_completion(messages)

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

                received_content = True

                yield content

            if not received_content:
                yield "I was unable to generate a response."

        except Exception:
            logger.exception("LLM streaming failed")
            raise

    def create_completion(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        client = self._require_client()

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=900,
            reasoning_effort="none",
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

        if not content:
            return ""

        return self._clean_model_output(content)

    def stream_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        history: List[Dict[str, str]],
        crag_decision: str = "UNKNOWN",
        long_term_memories: List[Dict] | None = None,
    ) -> Iterator[str]:
        generator = self._get_generator()

        yield from generator.stream(
            query=query,
            retrieved_chunks=retrieved_chunks,
            history=history,
            crag_decision=crag_decision,
            long_term_memories=long_term_memories or [],
        )

    def complete_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        history: List[Dict[str, str]],
        crag_decision: str = "UNKNOWN",
        long_term_memories: List[Dict] | None = None,
    ) -> str:
        generator = self._get_generator()

        return generator.complete(
            query=query,
            retrieved_chunks=retrieved_chunks,
            history=history,
            crag_decision=crag_decision,
            long_term_memories=long_term_memories or [],
        )

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