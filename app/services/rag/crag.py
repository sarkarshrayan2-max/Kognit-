import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from groq import Groq

from app.services.search.restricted_web import RestrictedWebSearch

logger = logging.getLogger("kognit.crag")

COURSE_DOMAIN_MAP = {
    "DBMS": "Database Management Systems",
    "COA": "Computer Organization and Architecture",
    "OS": "Operating Systems",
    "DSP": "Digital Signal Processing",
    "CIRCUITS": "Electronic Circuits and Network Analysis",
    "EMT": "Electromagnetic Theory and Transmission Lines",
    "IOT": "Internet of Things and Embedded Systems",
}

# Generic terms that appear across many course domains and therefore should
# never, by themselves, be treated as strong evidence that a retrieved chunk
# is actually relevant to the student's question. ("protocol", "system",
# "device" etc. show up in DBMS, OS, IoT, EMT material alike.)
GENERIC_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "for",
    "to", "and", "or", "what", "why", "how", "does", "do", "did", "that",
    "this", "with", "which", "can", "could", "explain", "define",
    "describe", "about", "please", "give", "example", "again", "it",
    "its", "be", "as", "by", "from", "using", "used", "use",
}


DOMAIN_SYSTEM_PROMPT = """You are the course-scope classifier for KOGNIT.

Your task is NOT to determine whether the retrieved documents answer the question completely.

Your task is to determine whether the student's question is allowed to use external web sources under the selected course.

There are three possible classifications:

IN_SCOPE:
The question is clearly relevant to the selected course OR the retrieved course material clearly concerns the same topic.

AMBIGUOUS:
The question could reasonably belong to the selected course but its exact scope is unclear.

OUT_OF_SCOPE:
The question is clearly unrelated to the selected course and the retrieved course material does not establish a meaningful connection.

IMPORTANT RULES:

1. Retrieved course material has priority over generic assumptions about what a course normally contains.
2. If the retrieved course material clearly discusses the same topic as the question, classify IN_SCOPE even if the topic is unusual for the course.
3. A selected course may contain papers, assignments, supplementary material, or topics that are not obvious from the course name.
4. Do not reject a question merely because it is advanced.
5. Generic concepts should be classified as AMBIGUOUS when they could reasonably be relevant.
6. Clearly unrelated subjects must be OUT_OF_SCOPE.
7. SQL-specific questions such as INNER JOIN, GROUP BY, normalization, SQL queries, and relational database operations are DBMS topics.
8. Transformer concepts such as self-attention, positional encoding, multi-head attention, encoder-decoder architecture, embeddings, and attention mechanisms can be IN_SCOPE when retrieved course material contains those topics.

Return ONLY JSON:

{
  "classification": "IN_SCOPE" | "AMBIGUOUS" | "OUT_OF_SCOPE"
}
"""

class CRAGEvaluator:

    def __init__(
        self,
        high_threshold: float = 0.65,
        low_threshold: float = 0.20,
        strong_semantic_threshold: float = 0.50,
        min_lexical_overlap: float = 0.25,
        min_corroborating_chunks: int = 2,
    ):
        load_dotenv()

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

        # A chunk scoring between `strong_semantic_threshold` and
        # `high_threshold` is not auto-accepted on its own, but if enough
        # chunks independently clear it *and* share real vocabulary with the
        # query, that corroboration is treated as strong evidence too. This
        # effectively lowers the bar for genuinely strong semantic matches
        # without opening the door to single lucky embeddings.
        self.strong_semantic_threshold = strong_semantic_threshold
        self.min_lexical_overlap = min_lexical_overlap
        self.min_corroborating_chunks = min_corroborating_chunks

        self.web_searcher = RestrictedWebSearch()

        api_key = os.getenv("GROQ_API_KEY")

        self.client = (
            Groq(api_key=api_key)
            if api_key
            else None
        )

        self.model_name = os.getenv(
            "GROQ_MODEL",
            "qwen/qwen3.6-27b",
        )

    @staticmethod
    def _course_chunks(
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        normalized = []

        for chunk in chunks:
            item = dict(chunk)
            metadata = dict(
                item.get(
                    "metadata",
                    {},
                )
            )
            metadata["source_type"] = "course"
            item["metadata"] = metadata
            normalized.append(item)

        return normalized

    @staticmethod
    def _web_chunks(
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        normalized = []

        for chunk in chunks:
            item = dict(chunk)
            metadata = dict(
                item.get(
                    "metadata",
                    {},
                )
            )
            metadata["source_type"] = "web"
            item["metadata"] = metadata
            normalized.append(item)

        return normalized

    @staticmethod
    def _score(
        chunk: Dict[str, Any]
    ) -> float:

        if "rerank_score" in chunk:
            return float(
                chunk["rerank_score"]
            )

        return float(
            chunk.get(
                "score",
                0.0,
            )
        )

    @staticmethod
    def _course_description(
        course_code: Optional[str],
    ) -> str:

        normalized = (
            str(course_code or "")
            .strip()
            .upper()
        )

        return COURSE_DOMAIN_MAP.get(
            normalized,
            normalized,
        )

    @staticmethod
    def _keywords(text: str) -> Set[str]:
        """Extract meaningful, lowercased tokens from text, dropping short
        words and generic stopwords that carry no discriminating signal."""

        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]*", text.lower())

        return {
            token
            for token in tokens
            if len(token) > 2 and token not in GENERIC_STOPWORDS
        }

    def _lexical_overlap(
        self,
        query: str,
        chunk_text: str,
    ) -> float:
        """Fraction of the query's meaningful keywords that also appear in
        the chunk text. This is a cheap guard against chunks that only
        share generic vocabulary (e.g. "protocol", "system") with the
        query but are not actually about the same topic."""

        query_keywords = self._keywords(query)

        if not query_keywords:
            # Nothing meaningful to compare against — don't let this
            # silently count as strong evidence.
            return 0.0

        chunk_keywords = self._keywords(chunk_text)

        overlap = query_keywords & chunk_keywords

        return len(overlap) / len(query_keywords)

    def _has_strong_local_evidence(
        self,
        query: str,
        course_chunks: List[Dict[str, Any]],
    ) -> bool:
        """Decide whether retrieved course chunks constitute strong enough
        evidence to answer directly (CORRECT) without any scope check or
        web fallback.

        This requires more than a single high similarity score:
        - The top chunk must clear `high_threshold` AND share enough
          actual vocabulary with the query (not just generic terms), OR
        - Several chunks must independently clear a lower
          `strong_semantic_threshold` while each still sharing real
          vocabulary with the query — corroboration across chunks is
          treated as equivalent evidence to one very high scoring chunk.
        """

        if not course_chunks:
            return False

        top_chunk = course_chunks[0]
        top_score = self._score(top_chunk)
        top_text = str(top_chunk.get("text", ""))
        top_overlap = self._lexical_overlap(query, top_text)

        if (
            top_score >= self.high_threshold
            and top_overlap >= self.min_lexical_overlap
        ):
            return True

        supporting_chunks = [
            chunk
            for chunk in course_chunks[:5]
            if self._score(chunk) >= self.strong_semantic_threshold
            and self._lexical_overlap(
                query,
                str(chunk.get("text", "")),
            )
            >= self.min_lexical_overlap
        ]

        return len(supporting_chunks) >= self.min_corroborating_chunks

    @staticmethod
    def _extract_json(
        text: str,
    ) -> Dict[str, Any]:

        cleaned = text.strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "No JSON object found"
            )

        return json.loads(
            cleaned[start:end + 1]
        )

    def _classify_scope(
        self,
        query: str,
        course_code: Optional[str],
        local_chunks: List[Dict[str, Any]],
    ) -> str:

        course_description = (
            self._course_description(
                course_code
            )
        )

        excerpts = []

        for index, chunk in enumerate(
            local_chunks[:5],
            start=1,
        ):
            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            score = self._score(chunk)

            excerpts.append(
                f"CHUNK {index} "
                f"(retrieval score={score:.4f}):\n"
                f"{text[:1800]}"
            )

        retrieved_context = (
            "\n\n".join(excerpts)
            if excerpts
            else "No course material was retrieved."
        )

        if not self.client:
            return self._fallback_scope(
                query=query,
                course_code=course_code,
                local_chunks=local_chunks,
            )

        user_prompt = f"""SELECTED COURSE:
{course_code}

COURSE DESCRIPTION:
{course_description}

STUDENT QUESTION:
{query}

RETRIEVED COURSE MATERIAL:
{retrieved_context}

Classify the scope of the question.
Return ONLY the JSON object."""

        try:
            response = (
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": DOMAIN_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    temperature=0.0,
                    max_completion_tokens=100,
                    reasoning_effort="none",
                    response_format={
                        "type": "json_object",
                    },
                )
            )

            if not response.choices:
                return self._fallback_scope(
                    query,
                    course_code,
                    local_chunks,
                )

            content = (
                response.choices[0]
                .message
                .content
            )

            if not content:
                return self._fallback_scope(
                    query,
                    course_code,
                    local_chunks,
                )

            data = self._extract_json(
                content
            )

            classification = str(
                data.get(
                    "classification",
                    "OUT_OF_SCOPE",
                )
            ).upper().strip()

            if classification not in {
                "IN_SCOPE",
                "AMBIGUOUS",
                "OUT_OF_SCOPE",
            }:
                classification = (
                    "OUT_OF_SCOPE"
                )

            logger.info(
                "Scope classification=%s course=%s query=%s",
                classification,
                course_code,
                query,
            )

            return classification

        except Exception:
            logger.exception(
                "Scope classification failed"
            )

            return self._fallback_scope(
                query,
                course_code,
                local_chunks,
            )

    def _fallback_scope(
        self,
        query: str,
        course_code: Optional[str],
        local_chunks: List[Dict[str, Any]],
    ) -> str:

        if local_chunks:
            top_score = max(
                self._score(chunk)
                for chunk in local_chunks
            )

            if top_score >= self.low_threshold:
                return "IN_SCOPE"

        normalized_course = str(
            course_code or ""
        ).upper()

        query_lower = query.lower()

        course_keywords = {
            "DBMS": {
                "sql",
                "database",
                "dbms",
                "query",
                "join",
                "normalization",
                "transaction",
                "index",
                "relational",
            },
            "IOT": {
                "iot",
                "mqtt",
                "sensor",
                "actuator",
                "embedded",
                "protocol",
                "edge",
                "device",
                "internet of things",
            },
            "COA": {
                "processor",
                "cpu",
                "cache",
                "pipeline",
                "instruction",
                "architecture",
                "memory",
            },
            "OS": {
                "operating system",
                "process",
                "thread",
                "deadlock",
                "scheduling",
                "memory management",
                "virtual memory",
            },
            "DSP": {
                "signal",
                "sampling",
                "fourier",
                "filter",
                "fft",
                "dsp",
                "frequency",
            },
            "CIRCUITS": {
                "circuit",
                "diode",
                "transistor",
                "amplifier",
                "voltage",
                "current",
            },
            "EMT": {
                "electromagnetic",
                "wave",
                "transmission line",
                "antenna",
                "field",
                "maxwell",
            },
        }

        keywords = course_keywords.get(
            normalized_course,
            set(),
        )

        if any(
            keyword in query_lower
            for keyword in keywords
        ):
            return "IN_SCOPE"

        return "OUT_OF_SCOPE"

    def evaluate_and_route(
        self,
        query: str,
        local_chunks: List[Dict[str, Any]],
        course_code: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:

        course_chunks = self._course_chunks(
            local_chunks
        )

        course_chunks.sort(
            key=self._score,
            reverse=True,
        )

        if course_chunks and self._has_strong_local_evidence(
            query,
            course_chunks,
        ):
            logger.info(
                "Strong local evidence found for query=%s course=%s",
                query,
                course_code,
            )

            return (
                "CORRECT",
                course_chunks[:3],
            )

        scope = self._classify_scope(
            query=query,
            course_code=course_code,
            local_chunks=course_chunks,
        )

        if scope == "OUT_OF_SCOPE":
            logger.info(
                "Blocking external search for out-of-scope query: %s",
                query,
            )

            return (
                "OUT_OF_SCOPE",
                [],
            )

        if course_chunks:
            top_score = self._score(
                course_chunks[0]
            )

            if top_score >= self.low_threshold:
                web_results = (
                    self.web_searcher.search(
                        query=query,
                        course_code=course_code,
                        max_results=2,
                    )
                )

                routed_context = (
                    course_chunks[:2]
                )

                if web_results:
                    routed_context.extend(
                        self._web_chunks(
                            web_results
                        )
                    )

                if web_results:
                    return (
                        "WEB_FALLBACK",
                        routed_context,
                    )

                return (
                    "INSUFFICIENT",
                    course_chunks[:2],
                )

        web_results = self.web_searcher.search(
            query=query,
            course_code=course_code,
            max_results=3,
        )

        if web_results:
            return (
                "WEB_FALLBACK",
                self._web_chunks(
                    web_results
                ),
            )

        return (
            "NOT_FOUND",
            [],
        )