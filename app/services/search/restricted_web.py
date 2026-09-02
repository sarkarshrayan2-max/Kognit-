import os
from typing import Any, Dict, List
from tavily import TavilyClient

APPROVED_DOMAINS = [
    "geeksforgeeks.org",
    "nptel.ac.in",
    "allaboutcircuits.com",
    "tutorialspoint.com",
    "electronics-tutorials.ws",
]


class RestrictedWebSearch:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=api_key) if api_key else None

    def search(self, query: str, max_results: int = 2) -> List[Dict[str, Any]]:
        if not self.client:
            return []

        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                include_domains=APPROVED_DOMAINS,
                search_depth="basic",
            )
            web_chunks = []
            for item in response.get("results", []):
                web_chunks.append(
                    {
                        "score": 0.50,  # Baseline fallback score
                        "text": item.get("content", ""),
                        "metadata": {
                            "source": item.get("url", "Web Search"),
                            "page": "Web",
                            "course_code": "EXTERNAL",
                            "is_web": True,
                        },
                    }
                )
            return web_chunks
        except Exception as e:
            print(f"[-] Web search fallback failed: {e}")
            return []