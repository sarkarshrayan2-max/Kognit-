from app.services.retrieval.fusion import HybridRetriever


retriever = HybridRetriever()

results = retriever.search(
    query="What is food aditive?",
    course_code="COA",
    top_k=3,
)

print("\nRESULTS:")
print("=" * 60)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}")
    print("Score:", result.get("score"))

    print(
        "Text:",
        result.get("text", "")[:500]
    )

    print(
        "Metadata:",
        result.get("metadata", {})
    )