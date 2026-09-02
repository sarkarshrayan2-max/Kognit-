import os
import re
from typing import Any, Dict, Iterator, List
from dotenv import load_dotenv
from groq import Groq
from app.services.llm.prompts import SYSTEM_TEACHER_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()


class LLMGateway:
    def __init__(self, model_name: str = "qwen/qwen3.6-27b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def stream_answer(
        self, query: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> Iterator[str]:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            src = chunk.get("metadata", {}).get("source", "Unknown")
            page = chunk.get("metadata", {}).get("page", "?")
            context_parts.append(
                f"[Source {i+1}: {src}, Page {page}]\n{chunk.get('text', '')}"
            )
        formatted_context = "\n\n".join(context_parts)

        user_message = USER_PROMPT_TEMPLATE.format(
            context_chunks=formatted_context,
            query=query,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_TEACHER_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=2048,
            stream=True,
        )

        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta