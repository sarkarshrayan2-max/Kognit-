import asyncio
import json

from app.graph.workflow import kognit_graph


async def run_conversation_test():

    print("\n")
    print("=" * 70)
    print("KOGNIT CONVERSATIONAL LANGGRAPH TEST")
    print("=" * 70)

    session_history = []

    queries = [
        "What is a food additive?",
        "What are its main purposes?",
        "Which authority regulates it in India?",
        "What does that authority do?",
    ]

    for number, query in enumerate(queries, start=1):

        print("\n")
        print("#" * 70)
        print(f"QUERY {number}: {query}")
        print("#" * 70)

        initial_state = {
            "query": query,
            "course_code": "COA",
            "history": session_history,
            "top_k": 3,
        }

        try:
            result = await kognit_graph.ainvoke(initial_state)

        except Exception as exc:
            print("\nGRAPH ERROR")
            print("-" * 70)
            print(type(exc).__name__)
            print(str(exc))
            break

        print("\nGRAPH EXECUTION SUCCESS")
        print("-" * 70)

        print("\nINTENT")
        print("-" * 70)
        print(result.get("intent", "N/A"))

        print("\nSTANDALONE QUERY")
        print("-" * 70)
        print(result.get("standalone_query", "N/A"))

        print("\nRESPONSE TYPE")
        print("-" * 70)
        print(result.get("response_type", "N/A"))

        print("\nCRAG DECISION")
        print("-" * 70)
        print(result.get("crag_decision", "N/A"))

        print("\nCITATIONS")
        print("-" * 70)

        citations = result.get("citations", [])

        if citations:
            print(json.dumps(citations, indent=2))
        else:
            print("No citations")

        print("\nFINAL ANSWER")
        print("=" * 70)

        answer = result.get("answer", "")

        if answer:
            print(answer)
        else:
            print("NO ANSWER GENERATED")

        print("=" * 70)

        # ---------------------------------------------------------
        # UPDATE CONVERSATION HISTORY
        # ---------------------------------------------------------

        session_history.append({
            "role": "user",
            "content": query,
        })

        session_history.append({
            "role": "assistant",
            "content": answer,
        })

        print("\nHISTORY")
        print("-" * 70)
        print(f"Messages stored: {len(session_history)}")

        for message in session_history:
            print(f"{message['role']}: {message['content'][:150]}")

    print("\n")
    print("=" * 70)
    print("CONVERSATIONAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_conversation_test())