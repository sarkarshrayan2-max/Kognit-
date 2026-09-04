from app.services.retrieval.fusion import HybridRetriever
from app.services.rag.crag import CRAGEvaluator


def print_results(title, results):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if not results:
        print("NO RESULTS")
        return

    for i, chunk in enumerate(results, 1):
        print(f"\nResult {i}")
        print("-" * 50)
        print("Score:", chunk.get("score"))

        metadata = chunk.get("metadata", {})

        print("Source:", metadata.get("source"))
        print("Course:", metadata.get("course_code"))
        print("Unit:", metadata.get("unit"))
        print("Page:", metadata.get("page"))
        print("Chunk:", metadata.get("chunk_index"))
        print("Source Type:", metadata.get("source_type"))

        text = chunk.get("text", "")
        print("Text:")
        print(text[:500])


def main():
    course_code = "COA"

    queries = [
        "What is a food additive?",
        "What are the main regulatory principles for food additives?",
        "What is the role of FSSAI in food additive regulation?",
        "What is an inner join?",
    ]

    print("\nKOGNIT CRAG TEST")
    print("=" * 70)

    retriever = HybridRetriever()
    crag = CRAGEvaluator()

    for query in queries:

        print("\n\n")
        print("#" * 70)
        print("QUERY:", query)
        print("#" * 70)

        # --------------------------------------------------
        # 1. Retrieve local course material
        # --------------------------------------------------

        local_chunks = retriever.search(
            query=query,
            course_code=course_code,
            top_k=5,
            candidate_limit=10,
        )

        print_results(
            "LOCAL RETRIEVAL",
            local_chunks,
        )

        # --------------------------------------------------
        # 2. CRAG evaluation
        # --------------------------------------------------

        decision, routed_context = crag.evaluate_and_route(
            query=query,
            local_chunks=local_chunks,
            course_code=course_code,
        )

        print("\n")
        print("=" * 70)
        print("CRAG DECISION")
        print("=" * 70)

        print("Decision:", decision)

        # --------------------------------------------------
        # 3. Final routed context
        # --------------------------------------------------

        print_results(
            "FINAL ROUTED CONTEXT",
            routed_context,
        )


if __name__ == "__main__":
    main()