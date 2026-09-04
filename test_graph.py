import asyncio
import json

from app.graph.workflow import kognit_graph


async def run_test():
    print("\n")
    print("=" * 70)
    print("KOGNIT LANGGRAPH END-TO-END TEST")
    print("=" * 70)

    queries = [
        "What is a food additive?",
        "What are the main regulatory principles for food additives?",
        "What is the role of FSSAI in food additive regulation?",
        "What is an inner join?",
    ]

    for query in queries:
        print("\n")
        print("#" * 70)
        print(f"QUERY: {query}")
        print("#" * 70)

        initial_state = {
            "query": query,
            "course_code": "COA",
            "history": [],
            "top_k": 3,
        }

        try:
            result = await kognit_graph.ainvoke(initial_state)

        except Exception as exc:
            print("\nGRAPH ERROR")
            print("-" * 70)
            print(type(exc).__name__)
            print(str(exc))
            continue

        print("\nGRAPH EXECUTION SUCCESS")
        print("-" * 70)

        # ---------------------------------------------------------
        # Intent
        # ---------------------------------------------------------

        print("\nINTENT")
        print("-" * 70)
        print(result.get("intent", "N/A"))

        # ---------------------------------------------------------
        # Standalone query
        # ---------------------------------------------------------

        print("\nSTANDALONE QUERY")
        print("-" * 70)
        print(result.get("standalone_query", "N/A"))

        # ---------------------------------------------------------
        # Response type
        # ---------------------------------------------------------

        print("\nRESPONSE TYPE")
        print("-" * 70)
        print(result.get("response_type", "N/A"))

        # ---------------------------------------------------------
        # CRAG decision
        # ---------------------------------------------------------

        print("\nCRAG DECISION")
        print("-" * 70)
        print(result.get("crag_decision", "N/A"))

        # ---------------------------------------------------------
        # Local chunks
        # ---------------------------------------------------------

        local_chunks = result.get("local_chunks", [])

        print("\nLOCAL RETRIEVAL")
        print("-" * 70)
        print(f"Chunks retrieved: {len(local_chunks)}")

        for i, chunk in enumerate(local_chunks, start=1):
            metadata = chunk.get("metadata", {})

            print(f"\nResult {i}")
            print("-" * 50)
            print(f"Score: {chunk.get('score', 'N/A')}")
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Course: {metadata.get('course_code', 'N/A')}")
            print(f"Unit: {metadata.get('unit', 'N/A')}")
            print(f"Page: {metadata.get('page', 'N/A')}")
            print(f"Chunk: {metadata.get('chunk_index', 'N/A')}")

            text = chunk.get("text", "")
            print(f"Text:\n{text[:500]}")

        # ---------------------------------------------------------
        # Final context
        # ---------------------------------------------------------

        final_context = result.get("final_context", [])

        print("\nFINAL CONTEXT")
        print("-" * 70)
        print(f"Chunks passed to LLM: {len(final_context)}")

        for i, chunk in enumerate(final_context, start=1):
            metadata = chunk.get("metadata", {})

            print(f"\nContext {i}")
            print("-" * 50)
            print(f"Score: {chunk.get('score', 'N/A')}")
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Source Type: {metadata.get('source_type', 'N/A')}")
            print(f"Page: {metadata.get('page', 'N/A')}")

            text = chunk.get("text", "")
            print(f"Text:\n{text[:500]}")

        # ---------------------------------------------------------
        # Citations
        # ---------------------------------------------------------

        citations = result.get("citations", [])

        print("\nCITATIONS")
        print("-" * 70)

        if citations:
            print(json.dumps(citations, indent=2))
        else:
            print("No citations")

        # ---------------------------------------------------------
        # FINAL ANSWER
        # ---------------------------------------------------------

        print("\nFINAL ANSWER")
        print("=" * 70)

        answer = result.get("answer", "")

        if answer:
            print(answer)
        else:
            print("NO ANSWER GENERATED")

        print("=" * 70)

        # ---------------------------------------------------------
        # Raw graph state keys
        # ---------------------------------------------------------

        print("\nGRAPH STATE KEYS")
        print("-" * 70)

        for key in result.keys():
            print(f"✓ {key}")

        print("\n")


if __name__ == "__main__":
    asyncio.run(run_test())