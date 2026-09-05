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

VALID_INTENTS = {"CONVERSATIONAL", "TECHNICAL", "OFF_TOPIC"}

CONDENSE_SYSTEM_PROMPT = """You are the conversational controller and search query formulator for KOGNIT, an academic AI assistant for Electronics and Computer Science engineering.

Analyze the conversation history and the student's latest turn.

Classify the latest turn as exactly one of:

1. CONVERSATIONAL

Pure greetings, acknowledgments, confirmations, closures, or casual filler with no actual topic or content.

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

Any actual academic or technical question, explanation request, example request, clarification, or follow-up question about engineering, computer science, or a related technical subject.

Examples:
- explain again
- why is that?
- give an example
- can you clarify?
- what is normalization?
- explain inner join
- why does self-attention work?
- what is positional encoding?

For TECHNICAL:
- intent = TECHNICAL
- standalone_query must be a complete, self-contained search query.

3. OFF_TOPIC

A real, substantive message (not a greeting or filler) that is about a subject with NO technical, engineering, or computer science content — general knowledge, literature, entertainment, sports, personal topics, or any other subject unrelated to the assistant's academic domain.

CONVERSATIONAL is filler with no topic at all. OFF_TOPIC is a genuine question or statement that DOES have a topic, but that topic is not technical/academic.

Examples:
- tell me about Oliver Twist
- who won the world cup last year?
- what's a good recipe for pasta?
- recommend me a movie
- what's the weather like today?

For OFF_TOPIC:
- intent = OFF_TOPIC
- standalone_query = the student's topic, restated plainly (not transformed into a technical question)

IMPORTANT:
Do not decide whether a TECHNICAL question belongs to the selected course — that is handled elsewhere.
Do not add course names merely because a course was selected.
Do not transform a technical question into an artificial course-specific question.
Do not force a genuinely off-topic, non-technical message into TECHNICAL just because it is not a greeting.

The standalone query for TECHNICAL should preserve the student's actual technical topic.

If the student asks:
"What is positional encoding?"

Return:
{"intent": "TECHNICAL", "standalone_query": "What is positional encoding?"}

If the student asks:
"Why is that?"

and the previous technical topic was positional encoding, return:
{"intent": "TECHNICAL", "standalone_query": "Why is positional encoding used in Transformer models?"}

If the student asks:
"explain again"

and the previous technical topic was inner join, return:
{"intent": "TECHNICAL", "standalone_query": "Explain inner join in DBMS with an example"}

If the student asks:
"tell me about Oliver Twist"

Return:
{"intent": "OFF_TOPIC", "standalone_query": "Oliver Twist"}

Return ONLY a valid JSON object.
Do not use markdown fences.
Do not include reasoning or commentary.

Required format:

{
  "intent": "CONVERSATIONAL" or "TECHNICAL" or "OFF_TOPIC",
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
    def _extract_json(text: str) -> Dict:
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

        return json.loads(
            cleaned[start:end + 1]
        )

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
                f"{role.capitalize()}: {content[:500]}"
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
        """Returns (intent, standalone_query_or_original_text).

        intent is one of "CONVERSATIONAL", "TECHNICAL", "OFF_TOPIC".
        Callers should route OFF_TOPIC the same way as OUT_OF_SCOPE from
        the CRAG evaluator (i.e. decline to search course/web material for
        it), rather than treating it as a TECHNICAL query.
        """

        query = query.strip()

        if not query:
            return "CONVERSATIONAL", query

        if self._is_obviously_conversational(query):
            logger.info(
                "Conversational intent detected: %s",
                query,
            )
            return "CONVERSATIONAL", query

        if not self.client:
            logger.warning(
                "GROQ_API_KEY not configured. Skipping query condensation."
            )
            return "TECHNICAL", query

        formatted_history = self._format_history(history)

        user_content = (
            f"Selected course: {course_code}\n\n"
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
                return "TECHNICAL", query

            raw = (
                response.choices[0]
                .message
                .content
            )

            if not raw:
                return "TECHNICAL", query

            data = self._extract_json(raw)

            intent = str(
                data.get(
                    "intent",
                    "TECHNICAL",
                )
            ).upper().strip()

            if intent not in VALID_INTENTS:
                intent = "TECHNICAL"

            if intent == "CONVERSATIONAL":
                return "CONVERSATIONAL", query

            standalone = data.get(
                "standalone_query"
            )

            if not standalone:
                standalone = query

            standalone = str(
                standalone
            ).strip()

            if not standalone:
                standalone = query

            if intent == "OFF_TOPIC":
                logger.info(
                    "Off-topic (non-technical) intent detected: %s",
                    query,
                )
                return "OFF_TOPIC", standalone

            return "TECHNICAL", standalone

        except Exception:
            logger.exception(
                "Query condensation failed"
            )

            return "TECHNICAL", query