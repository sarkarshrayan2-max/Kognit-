import json
import logging
import os
import re
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger("kognit.condenser")


CONVERSATIONAL_PATTERNS = {
    "ok",
    "okay",
    "thanks",
    "thank you",
    "got it",
    "understood",
    "hello",
    "hi",
    "hey",
    "bye",
    "done",
    "goodbye",
    "cool",
    "sure",
    "yup",
    "yes",
    "no",
    "alright",
    "fine",
    "good",
    "great",
    "nice",
    "welcome",
    "sup",
    "yo",
    "lol",
    "haha",
    "lmao",
    "lolz",
    "hehe",
    "rofl",
    "roflmao",
    "lmfao",
    "lmfao!!",
    "that works",
    "all set",
    "perfect",
    "finished",
}


CONDENSE_SYSTEM_PROMPT = """You are the conversational controller and search query formulator for KOGNIT, an academic AI assistant for Electronics and Computer Science (ECS) engineering.

Analyze the conversation history and the student's latest turn.

Classify the latest turn as either:

1. CONVERSATIONAL
Pure greetings, acknowledgments, confirmations, closures, or casual filler.

Examples:
- ok
- okay
- thanks
- got it
- understood
- cool
- done
- all set
- perfect
- goodbye

For CONVERSATIONAL:
- intent = CONVERSATIONAL
- standalone_query = null

2. TECHNICAL
Any actual academic or technical question, explanation request, example request, clarification, or follow-up question.

Examples:
- explain again
- why is that?
- give an example
- can you clarify?
- what is normalization?
- explain inner join

For TECHNICAL:
- intent = TECHNICAL
- standalone_query must be a complete, self-contained search query.

Use conversation history to identify the actual technical topic.

IMPORTANT:
Never use conversational filler as the technical subject.
Words such as:
- ok
- cool
- done
- yes
- please
- again
- thanks

must not become the search topic.

Example:

History:
User: What is an Inner Join?
Assistant: An Inner Join combines matching rows.

Latest Turn:
explain again

Correct standalone query:
Inner Join definition mechanism and examples in DBMS

Return ONLY a valid JSON object.
Do not use markdown fences.
Do not include reasoning, commentary, or any text before or after the JSON.

Required format:

{
  "intent": "CONVERSATIONAL" or "TECHNICAL",
  "standalone_query": "string or null"
}
"""


class QueryCondenser:

    def __init__(
        self,
        model_name: str = "qwen/qwen3.6-27b",
    ):
        api_key = os.getenv("GROQ_API_KEY")

        self.client = (
            Groq(api_key=api_key)
            if api_key
            else None
        )

        self.model_name = model_name



    def _is_obviously_conversational(
        self,
        query: str,
    ) -> bool:

        normalized = query.strip().lower()

        if normalized in CONVERSATIONAL_PATTERNS:
            return True

        cleaned = re.sub(
            r"[^a-z\s]",
            "",
            normalized,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        return cleaned in CONVERSATIONAL_PATTERNS

    

    @staticmethod
    def _extract_json(
        text: str,
    ) -> Dict:

        cleaned = text.strip()

        cleaned = re.sub(
            r"^```json\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"^```\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        cleaned = cleaned.strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "No JSON object found in condenser response"
            )

        json_text = cleaned[
            start : end + 1
        ]

        return json.loads(json_text)

    

    @staticmethod
    def _format_history(
        history: List[Dict[str, str]],
    ) -> str:

        if not history:
            return "No previous context."

        history_parts = []

        for message in history[-4:]:

            role = message.get(
                "role",
                "user",
            )

            content = message.get(
                "content",
                "",
            ).strip()

            if not content:
                continue

            history_parts.append(
                f"{role.capitalize()}: "
                f"{content[:500]}"
            )

        if not history_parts:
            return "No previous context."

        return "\n".join(history_parts)

    

    def analyze(
        self,
        query: str,
        history: List[Dict[str, str]],
        course_code: str,
    ) -> Tuple[str, str]:

        query = query.strip()

        

        if not query:

            return (
                "CONVERSATIONAL",
                query,
            )

        

        if self._is_obviously_conversational(
            query
        ):

            logger.info(
                "Conversational intent detected: %s",
                query,
            )

            return (
                "CONVERSATIONAL",
                query,
            )

        

        if not self.client:

            logger.warning(
                "GROQ_API_KEY not configured. "
                "Skipping query condensation."
            )

            return (
                "TECHNICAL",
                query,
            )

        

        formatted_history = (
            self._format_history(history)
        )

        

        user_content = (
            f"Course: {course_code}\n\n"
            f"Conversation History:\n"
            f"{formatted_history}\n\n"
            f"Latest Student Turn:\n"
            f"{query}\n\n"
            "Return ONLY the required JSON object."
        )

        messages = [
            {
                "role": "system",
                "content": CONDENSE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        

        try:

            response = (
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_completion_tokens=200,
                    reasoning_effort="none",
                    response_format={
                        "type": "json_object",
                    },
                )
            )

            if not response.choices:

                logger.warning(
                    "Condenser returned no choices."
                )

                return (
                    "TECHNICAL",
                    query,
                )

            raw_text = (
                response.choices[0]
                .message
                .content
                or ""
            ).strip()

            if not raw_text:

                logger.warning(
                    "Condenser returned empty content."
                )

                return (
                    "TECHNICAL",
                    query,
                )

            

            try:
                data = self._extract_json(raw_text)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Invalid condenser JSON: %s. Using original query.",
                    exc,
                )

                return (
                    "TECHNICAL",
                    query,
                )

            

            if not isinstance(data, dict):
                logger.warning(
                    "Condenser JSON is not an object: %r",
                    type(data).__name__,
                )
                return (
                    "TECHNICAL",
                    query,
                )

            intent = str(
                data.get(
                    "intent",
                    "TECHNICAL",
                )
            ).upper().strip()

            if intent not in {
                "CONVERSATIONAL",
                "TECHNICAL",
            }:

                logger.warning(
                    "Invalid condenser intent: %s",
                    intent,
                )

                intent = "TECHNICAL"

            

            if intent == "CONVERSATIONAL":

                return (
                    "CONVERSATIONAL",
                    query,
                )

            

            standalone_query = data.get(
                "standalone_query"
            )

            if not isinstance(
                standalone_query,
                str,
            ):

                standalone_query = query

            standalone_query = (
                standalone_query.strip()
            )

            if not standalone_query:

                standalone_query = query

            logger.info(
                "Condensed query: '%s' -> '%s'",
                query,
                standalone_query,
            )

            return (
                "TECHNICAL",
                standalone_query,
            )

        

        except Exception as exc:

            logger.exception(
                "Query condensation failed: %s",
                exc,
            )

            

            if history:

                last_technical_turn = next(
                    (
                        message.get(
                            "content",
                            "",
                        )
                        for message in reversed(
                            history
                        )
                        if (
                            message.get(
                                "role"
                            )
                            == "user"
                            and not self._is_obviously_conversational(
                                message.get(
                                    "content",
                                    "",
                                )
                            )
                        )
                    ),
                    None,
                )

                current_words = query.split()

                if (
                    last_technical_turn
                    and 0 < len(current_words) <= 4
                ):

                    fallback_query = (
                        f"{last_technical_turn} "
                        "detailed explanation and examples"
                    )

                    logger.info(
                        "Using condenser fallback: %s",
                        fallback_query,
                    )

                    return (
                        "TECHNICAL",
                        fallback_query,
                    )

            return (
                "TECHNICAL",
                query,
            )