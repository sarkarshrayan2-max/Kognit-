import os
from typing import List, Dict, Tuple
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

INTENT_AND_CONDENSE_PROMPT = """You are the conversational controller for KOGNIT, an academic assistant for Electronics and Computer Science (ECS).

Given the recent chat history and the student's latest message:
1. Classify the intent into one of two types:
   - "CONVERSATIONAL": Acknowledgements, greetings, filler ("ok", "okay", "thanks", "got it", "cool", "hello", "understood").
   - "TECHNICAL": Questions, requests for examples, requests to re-explain a topic, or follow-ups.
2. If CONVERSATIONAL, output:
   INTENT: CONVERSATIONAL
   QUERY: NONE
3. If TECHNICAL, rewrite the query into an independent technical search query targeting the course "{course_code}".
   - If the user says "explain again", "elaborate", or "give an example", find the LAST technical subject discussed in the history and form: "[Subject] detailed explanation and examples".
   - Never search for literal terms like "ok", "explain", or "again".
   Output:
   INTENT: TECHNICAL
   QUERY: <standalone query here>

Course: {course_code}

Chat History:
{chat_history}

Student Question: {question}
"""

class QueryCondenser:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None
        self.model_name = model_name

    def analyze(self, query: str, history: List[Dict[str, str]], course_code: str) -> Tuple[str, str]:
        """
        Returns: (intent, rewritten_query)
        intent is either 'CONVERSATIONAL' or 'TECHNICAL'
        """
        cleaned_query = query.strip().lower()

        
        trivial_pleasantries = {"ok", "okay", "k", "thanks", "thank you", "got it", "understood", "cool", "great", "nice"}
        if cleaned_query in trivial_pleasantries:
            return "CONVERSATIONAL", query

        if not self.client:
            return "TECHNICAL", query

        formatted_history = "No previous context."
        if history:
            formatted_history = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-4:]]
            )

        prompt = INTENT_AND_CONDENSE_PROMPT.format(
            course_code=course_code,
            chat_history=formatted_history,
            question=query
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            raw_output = res.choices[0].message.content.strip()

            intent = "TECHNICAL"
            standalone = query

            for line in raw_output.splitlines():
                if line.startswith("INTENT:"):
                    intent = line.replace("INTENT:", "").strip()
                elif line.startswith("QUERY:"):
                    extracted = line.replace("QUERY:", "").strip()
                    if extracted and extracted != "NONE":
                        standalone = extracted

            return intent, standalone
        except Exception:
            return "TECHNICAL", query