import json
import logging
import os
import re
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger("kognit.condenser")

CONDENSE_SYSTEM_PROMPT = """You are the conversational controller and search query formulator for KOGNIT, an academic AI assistant for Electronics and Computer Science (ECS) engineering.

Analyze the conversation history and the student's latest turn to classify intent and formulate a standalone query.

### Classification Rules:
1. "CONVERSATIONAL": Pure acknowledgments, greetings, closures, or casual filler (e.g., "ok", "thanks", "got it", "hello", "understood").
   - Set "intent": "CONVERSATIONAL"
   - Set "standalone_query": null

2. "TECHNICAL": The student is asking a technical question, requesting an example, or asking for clarification/re-explanation (e.g., "explain again", "why is that?", "give an example", "can you clarify?").
   - Set "intent": "TECHNICAL"
   - Set "standalone_query": Formulate a complete, self-contained search query. Identify the technical topic from the immediate conversation history and state it clearly with relevant engineering terms. Never output conversational words ("again", "explain", "please") as the query.

### Examples:
Example 1:
Input:
Course: DBMS
History:
User: What is an Inner Join?
Assistant: An Inner Join combines matching rows...
Latest Turn: explain again
Output:
{"intent": "TECHNICAL", "standalone_query": "Inner Join definition mechanism and examples in DBMS"}

Example 2:
Input:
Course: DBMS
History:
User: What is universal quantifiers, existential quantifier and free and bound variable?
Assistant: The universal quantifier (forall) requires all tuples...
Latest Turn: can you give an example of the first one?
Output:
{"intent": "TECHNICAL", "standalone_query": "Universal quantifier in relational calculus detailed examples and syntax"}

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
You MUST respond with a valid JSON object ONLY. Do not write any preamble, Markdown ticks, or explanations.
{
  "intent": "CONVERSATIONAL" | "TECHNICAL",
  "standalone_query": "string or null"
}"""


class QueryCondenser:
    def __init__(self, model_name: str = "qwen/qwen3.6-27b"):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None
        self.model_name = model_name

    def analyze(
        self, query: str, history: List[Dict[str, str]], course_code: str
    ) -> Tuple[str, str]:
        """
        Dynamically determines intent and produces a standalone query.
        Returns: (intent, standalone_query)
        """
        if not self.client:
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
            standalone = data.get("standalone_query") or query
            return intent, standalone

        except Exception as e:
            logger.warning("Groq JSON response parsing failed (%s). Attempting regex extraction.", e)
            # Fallback: extract JSON with regex if markdown backticks were returned
            try:
                if 'raw_text' in locals() and raw_text:
                    match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        return data.get("intent", "TECHNICAL"), data.get("standalone_query") or query
            except Exception:
                pass

            if history:
                last_user_turn = next((m["content"] for m in reversed(history) if m.get("role") == "user"), None)
                if last_user_turn and len(query.strip().split()) <= 4:
                    fallback_standalone = f"{last_user_turn} detailed explanation and examples"
                    logger.info("Failsafe recovery used: '%s'", fallback_standalone)
                    return "TECHNICAL", fallback_standalone

            return "TECHNICAL", query