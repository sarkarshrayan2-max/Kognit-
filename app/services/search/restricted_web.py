import os
from typing import Any, Dict, List, Optional
from tavily import TavilyClient

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
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=api_key) if api_key else None

    def search(
        self,
        query: str,
        course_code: Optional[str] = None,
        max_results: int = 2
    ) -> List[Dict[str, Any]]:
        if not self.client:
            return []

        # Anchor ambiguous queries and acronyms to the academic syllabus domain
        anchored_query = query.strip()
        if course_code and course_code in COURSE_DOMAIN_MAP:
            domain_context = COURSE_DOMAIN_MAP[course_code]
            # Avoid duplicating course prefix if the query already specifies it
            if course_code.lower() not in anchored_query.lower():
                anchored_query = f"{domain_context}: {anchored_query}"
            else:
                anchored_query = f"Engineering {domain_context} {anchored_query}"

        try:
            response = self.client.search(
                query=anchored_query,
                max_results=max_results,
                include_domains=APPROVED_DOMAINS,
                search_depth="basic",
            )
            web_chunks = []
            for item in response.get("results", []):
                content = item.get("content", "").strip()
                if not content:
                    continue

                web_chunks.append(
                    {
                        "score": 0.50,
                        "text": content,
                        "metadata": {
                            "source": item.get("url", "Web Search"),
                            "page": "Web",
                            "course_code": course_code or "EXTERNAL",
                            "is_web": True,
                        },
                    }
                )
            return web_chunks
        except Exception as e:
            print(f"[-] Web search fallback failed for query '{anchored_query}': {e}")
            return []