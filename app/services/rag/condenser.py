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

Analyze the conversation history and the student's latest turn to classify intent and formulate a standalone query.

### Classification Rules:

1. "CONVERSATIONAL":
   Pure acknowledgments, greetings, closures, status confirmations, or casual filler.
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

   Set:
   "intent": "CONVERSATIONAL"
   "standalone_query": null

2. "TECHNICAL":
   The student is asking an actual technical question, requesting an example, or asking for clarification/re-explanation of an academic topic.
   Examples:
   - explain again
   - why is that?
   - give an example
   - can you clarify?
   - what is normalization?
   - explain inner join

   Set:
   "intent": "TECHNICAL"

   For TECHNICAL queries, formulate a complete, self-contained search query.
   - Use the conversation history to identify the technical topic.
   - CRITICAL: Never use conversational words or filler (e.g., "cool", "done", "ok", "yes", "please", "again") as the subject of the search query. Always extract the real underlying engineering subject from context.
   - Include core concepts, relevant terminology, and course context.

### Examples:

Example 1:
Input:
Course: DBMS
History:
User: cool
Assistant: Understood! Let me know if you want to explore more examples or dive into another topic.
Latest Turn: done

Output:
{"intent": "CONVERSATIONAL", "standalone_query": null}

Example 2:
Input:
Course: DBMS
History:
User: What is an Inner Join?
Assistant: An Inner Join combines matching rows...
Latest Turn: explain again

Output:
{"intent": "TECHNICAL", "standalone_query": "Inner Join definition mechanism and examples in DBMS"}

Example 3:
Input:
Course: COA
History:
User: How does Booth's algorithm handle negative multipliers?
Assistant: Booth's algorithm examines bit pairs...
Latest Turn: thanks got it!

Output:
{"intent": "CONVERSATIONAL", "standalone_query": null}

### Strict Output Requirement:
You MUST respond with a valid JSON object ONLY.
Do not write any preamble, Markdown ticks, or explanations.
{
  "intent": "CONVERSATIONAL" | "TECHNICAL",
  "standalone_query": "string or null"
}
"""


class QueryCondenser:
    def __init__(self, model_name: str = "qwen/qwen3.6-27b"):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None
        self.model_name = model_name

    def _is_obviously_conversational(self, query: str) -> bool:
        """
        Detect obvious conversational messages without calling the LLM.
        Prevents conversational fillers from entering technical retrieval flows.
        """
        normalized = query.strip().lower()

        if normalized in CONVERSATIONAL_PATTERNS:
            return True

        cleaned = re.sub(r"[^a-z\s]", "", normalized)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if cleaned in CONVERSATIONAL_PATTERNS:
            return True

        return False

    def analyze(
        self,
        query: str,
        history: List[Dict[str, str]],
        course_code: str,
    ) -> Tuple[str, str]:
        """
        Dynamically determines intent and produces a standalone query.
        Returns:
            (intent, standalone_query)
        """
        if self._is_obviously_conversational(query):
            logger.info("Deterministic conversational intent detected: '%s'", query)
            return "CONVERSATIONAL", query

        if not self.client:
            logger.warning("GROQ_API_KEY not configured. Using original query without condensation.")
            return "TECHNICAL", query

        formatted_history = "No previous context."
        if history:
            formatted_history = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content'][:300]}" for msg in history[-4:]]
            )

        user_content = (
            f"Course: {course_code}\n"
            f"History:\n{formatted_history}\n"
            f"Latest Turn: {query}"
        )

        messages = [
            {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"},
            )

            raw_text = res.choices[0].message.content.strip()
            data = json.loads(raw_text)

            intent = data.get("intent", "TECHNICAL")
            standalone = data.get("standalone_query")

            if intent not in {"CONVERSATIONAL", "TECHNICAL"}:
                logger.warning("Unexpected intent '%s' from condenser. Defaulting to TECHNICAL.", intent)
                intent = "TECHNICAL"

            if intent == "CONVERSATIONAL":
                return "CONVERSATIONAL", query

            standalone = standalone or query
            return "TECHNICAL", standalone

        except Exception as e:
            logger.warning("Groq JSON response parsing failed (%s). Attempting regex extraction.", e)

            try:
                if "raw_text" in locals() and raw_text:
                    match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        intent = data.get("intent", "TECHNICAL")
                        standalone = data.get("standalone_query")

                        if intent == "CONVERSATIONAL":
                            return "CONVERSATIONAL", query

                        return "TECHNICAL", standalone or query
            except Exception as regex_error:
                logger.warning("Regex JSON recovery also failed: %s", regex_error)

            if history:
                last_technical_turn = next(
                    (
                        m["content"]
                        for m in reversed(history)
                        if m.get("role") == "user"
                        and not self._is_obviously_conversational(m.get("content", ""))
                    ),
                    None,
                )

                current_words = query.strip().split()
                if last_technical_turn and 0 < len(current_words) <= 4:
                    fallback_standalone = f"{last_technical_turn} detailed explanation and examples"
                    logger.info("Failsafe recovery used: '%s'", fallback_standalone)
                    return "TECHNICAL", fallback_standalone

            return "TECHNICAL", query