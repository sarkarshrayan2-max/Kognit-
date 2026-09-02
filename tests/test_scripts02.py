import os
from dotenv import load_dotenv
from app.services.ingestion.indexer import DocumentIndexer
from app.services.retrieval.fusion import HybridRetriever
from app.services.llm.gateway import LLMGateway

load_dotenv()


print("[*] Performing Hybrid Search (Dense + BM25 + BGE Reranker)...")
retriever = HybridRetriever()
results = retriever.search(
    query="What is the meaning of food addictive ?",
    course_code="COA",
    top_k=3
)

if not results:
    print("[-] No relevant chunks retrieved. Check if Qdrant has indexed data.")
    exit()

print(f"[+] Retrieved {len(results)} chunks. Top score: {results[0]['score']:.4f}")


print("[*] Generating teacher-style answer via Groq...")
gateway = LLMGateway()
output = gateway.generate_answer(
    query="What is the meaning of food addictive ?",
    retrieved_chunks=results
)

print("\n--- KOGNIT ANSWER ---")
print(output["answer"])

print("\n--- CITATIONS ---")
for cite in output["citations"]:
    print(f"• {cite['source']} (Page {cite['page']}) - Reranker Score: {cite['score']}")