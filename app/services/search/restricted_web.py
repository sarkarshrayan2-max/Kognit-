import os
import logging
from typing import Any, Dict, List, Optional

from tavily import TavilyClient


logger = logging.getLogger("kognit.web_search")


APPROVED_DOMAINS = [
    "geeksforgeeks.org",
    "nptel.ac.in",
    "allaboutcircuits.com",
    "tutorialspoint.com",
    "electronics-tutorials.ws",
]


COURSE_DOMAIN_MAP = {
    "DBMS": "Database Management Systems (DBMS)",
    "COA": "Computer Organization and Architecture (COA)",
    "OS": "Operating Systems (OS)",
    "DSP": "Digital Signal Processing (DSP)",
    "CIRCUITS": "Electronic Circuits and Network Analysis",
    "EMT": "Electromagnetic Theory and Transmission Lines (EMT)",
    "IOT": "Internet of Things (IoT) Embedded Systems",
}


class RestrictedWebSearch:
    """
    Restricted external web search used only as a CRAG fallback.

    Web results are explicitly marked with:
        source_type = "web"

    This prevents external information from being confused
    with course-document information downstream.
    """

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")

        self.client = (
            TavilyClient(api_key=api_key)
            if api_key
            else None
        )

    def _build_query(
        self,
        query: str,
        course_code: Optional[str],
    ) -> str:
        """
        Add course context to ambiguous external searches.
        """

        anchored_query = query.strip()

        if not course_code:
            return anchored_query

        normalized_course = course_code.strip().upper()

        domain_context = COURSE_DOMAIN_MAP.get(normalized_course)

        if not domain_context:
            return anchored_query

        # Avoid unnecessarily repeating the course code.
        if normalized_course.lower() not in anchored_query.lower():
            return f"{domain_context}: {anchored_query}"

        return f"Engineering {domain_context}: {anchored_query}"

    def search(
        self,
        query: str,
        course_code: Optional[str] = None,
        max_results: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Search approved external domains.

        Returns results in the same structure expected by CRAG:

        {
            "score": ...,
            "text": ...,
            "metadata": {
                "source": ...,
                "url": ...,
                "page": "Web",
                "course_code": ...,
                "source_type": "web"
            }
        }
        """

        if not self.client:
            logger.warning(
                "TAVILY_API_KEY not configured. "
                "Skipping external web search."
            )
            return []

        if not query or not query.strip():
            return []

        max_results = max(1, min(max_results, 5))

        anchored_query = self._build_query(
            query=query,
            course_code=course_code,
        )

        try:
            response = self.client.search(
                query=anchored_query,
                max_results=max_results,
                include_domains=APPROVED_DOMAINS,
                search_depth="basic",
            )

            results = response.get("results", [])

            web_chunks: List[Dict[str, Any]] = []

            for item in results:
                content = (item.get("content") or "").strip()
                url = (item.get("url") or "").strip()
                title = (item.get("title") or "").strip()

                
                if not content:
                    continue

                web_chunks.append(
                    {
                        
                        "score": 0.0,

                        "text": content,

                        "metadata": {
                            "source": title or url or "Web Search",
                            "url": url,
                            "page": "Web",
                            "course_code": (
                                course_code.upper()
                                if course_code
                                else "EXTERNAL"
                            ),
                            "source_type": "web",
                            "is_web": True,
                        },
                    }
                )

            return web_chunks

        except Exception as exc:
            logger.error(
                "Web search fallback failed for query '%s': %s",
                anchored_query,
                exc,
                exc_info=True,
            )

            return []