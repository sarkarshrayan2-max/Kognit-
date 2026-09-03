import os
import re
from typing import Any, Dict, Iterator, List, Optional
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
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            src = chunk.get("metadata", {}).get("source", "Unknown")
            page = chunk.get("metadata", {}).get("page", "?")
            context_parts.append(
                f"[Source {i+1}: {src}, Page {page}]\n{chunk.get('text', '')}"
            )
        formatted_context = "\n\n".join(context_parts) if context_parts else "No context available."

        formatted_history = "No previous context."
        if history:
            formatted_history = "\n".join(
                [f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]]
            )

        user_message = USER_PROMPT_TEMPLATE.format(
            context_chunks=formatted_context,
            chat_history=formatted_history,
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

        inside_think_tag = False
        buffer = ""

        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue

            buffer += delta

            while True:
                if not inside_think_tag:
                    if "<think>" in buffer:
                        pre, match, post = buffer.partition("<think>")
                        if pre:
                            yield pre
                        buffer = post
                        inside_think_tag = True
                    else:
                        if buffer.startswith("Here's a thinking process:"):
                            if "\n\n" in buffer:
                                _, buffer = buffer.split("\n\n", 1)
                            else:
                                break
                        else:
                            yield buffer
                            buffer = ""
                        break
                else:
                    if "</think>" in buffer:
                        _, match, post = buffer.partition("</think>")
                        buffer = post
                        inside_think_tag = False
                    else:
                        buffer = "" 
                        break