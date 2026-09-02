from dotenv import load_dotenv
load_dotenv()

from app.services.ingestion.indexer import DocumentIndexer
from app.services.rag.retriever import kognit_graph

# 1. Index the new subject document
print("[*] Ingesting DBMS course material...")
indexer = DocumentIndexer()

# Make sure 'sample_dbms.pdf' exists in your root folder
chunks_indexed = indexer.index_pdf(
    pdf_path="sample_dbms.pdf",
    course_code="DBMS",
    unit=3,
    visibility="global"
)
print(f"[+] Successfully indexed {chunks_indexed} chunks for DBMS.\n")

# 2. Test Case A: Ask a DBMS question scoped to DBMS (Should succeed locally)
print("=" * 60)
print("TEST 1: In-domain query (DBMS question -> DBMS course)")
print("=" * 60)
res_dbms = kognit_graph.invoke({
    "query": "Explain Scheduling",
    "course_code": "DBMS",
    "top_k": 3,
    "local_chunks": [],
    "final_context": [],
    "crag_decision": "PENDING",
    "answer": "",
    "citations": [],
    "model_used": "",
})

print(f"CRAG Decision: {res_dbms['crag_decision']}")
print("Citations:")
for c in res_dbms["citations"]:
    print(f" - Source: {c['source']}, Page: {c['page']}, Score: {c['score']}")
print("\nAnswer Snippet:\n", res_dbms["answer"][:300], "...\n")


# 3. Test Case B: Ask a COA question scoped to DBMS (Should trigger isolation/CRAG fallback)
print("=" * 60)
print("TEST 2: Cross-domain isolation check (COA question -> DBMS course)")
print("=" * 60)
res_cross = kognit_graph.invoke({
    "query": "What is food additive ?",
    "course_code": "DBMS",  
    "top_k": 3,
    "local_chunks": [],
    "final_context": [],
    "crag_decision": "PENDING",
    "answer": "",
    "citations": [],
    "model_used": "",
})

print(f"CRAG Decision: {res_cross['crag_decision']}")
print("Citations:")
for c in res_cross["citations"]:
    print(f" - Source: {c['source']}, Page: {c['page']}, Score: {c['score']}")