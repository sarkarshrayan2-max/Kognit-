from app.services.ingestion.indexer import DocumentIndexer
from app.services.retrieval.fusion import HybridRetriever


indexer = DocumentIndexer()
total_chunks = indexer.index_pdf("Sample_COA.pdf", course_code="COA", unit=2)
print(f"Indexed {total_chunks} chunks.")


retriever = HybridRetriever()
results = retriever.search(
    query="what is the meaning of “Food Additive” ", course_code="COA", top_k=3
)

for r in results:
    print(f"Score: {r['score']:.4f} | Source: {r['metadata']['source']} (Page {r['metadata']['page']})")
    print(r["text"][:200] + "...\n")