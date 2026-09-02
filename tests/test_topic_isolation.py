from dotenv import load_dotenv
load_dotenv()  # Load environment keys before module initialization

from app.services.ingestion.indexer import DocumentIndexer
from app.services.rag.retriever import kognit_graph

def run_tests():
    indexer = DocumentIndexer()

    # 1. Ingest both PDF documents
    print("[*] Ingesting sample_coa.pdf into COA collection...")
    coa_chunks = indexer.index_pdf(
        pdf_path="Sample_COA.pdf",
        course_code="COA",
        unit=1,
        visibility="global"
    )
    print(f"[+] COA chunks indexed: {coa_chunks}")

    print("[*] Ingesting sample_dbms.pdf into DBMS collection...")
    dbms_chunks = indexer.index_pdf(
        pdf_path="sample_dbms.pdf",
        course_code="DBMS",
        unit=2,
        visibility="global"
    )
    print(f"[+] DBMS chunks indexed: {dbms_chunks}\n")

    # Helper function to run graph
    def ask_kognit(query: str, course: str):
        state = {
            "query": query,
            "course_code": course,
            "top_k": 3,
            "local_chunks": [],
            "final_context": [],
            "crag_decision": "PENDING",
            "answer": "",
            "citations": [],
            "model_used": "",
        }
        return kognit_graph.invoke(state)

    # -------------------------------------------------------------
    # TEST 1: COA Query under COA
    # -------------------------------------------------------------
    print("=" * 65)
    print("TEST 1: In-domain COA (Question: Pipelining hazards | Filter: COA)")
    print("=" * 65)
    res_coa = ask_kognit("Explain food additive", "COA")
    print(f"CRAG Decision: {res_coa['crag_decision']}")
    print("Citations retrieved:")
    for c in res_coa["citations"]:
        print(f"  • Source: {c['source']} | Page: {c['page']} | Score: {c['score']}")
    print("\nAnswer Snippet:\n", res_coa["answer"][:250], "...\n")

    # -------------------------------------------------------------
    # TEST 2: DBMS Query under DBMS
    # -------------------------------------------------------------
    print("=" * 65)
    print("TEST 2: In-domain DBMS (Question: Normalization / BCNF | Filter: DBMS)")
    print("=" * 65)
    res_dbms = ask_kognit("What is Scheduling", "DBMS")
    print(f"CRAG Decision: {res_dbms['crag_decision']}")
    print("Citations retrieved:")
    for c in res_dbms["citations"]:
        print(f"  • Source: {c['source']} | Page: {c['page']} | Score: {c['score']}")
    print("\nAnswer Snippet:\n", res_dbms["answer"][:250], "...\n")

    # -------------------------------------------------------------
    # TEST 3: Cross-Domain Leak Check
    # -------------------------------------------------------------
    print("=" * 65)
    print("TEST 3: Cross-domain Isolation Check (Question: COA topic | Filter: DBMS)")
    print("=" * 65)
    res_leak = ask_kognit("What is non scheduling", "COA")
    print(f"CRAG Decision: {res_leak['crag_decision']}")
    print("Citations retrieved:")
    for c in res_leak["citations"]:
        print(f"  • Source: {c['source']} | Page: {c['page']} | Score: {c['score']}")
    print("\nOutcome:")
    if any("coa" in str(c["source"]).lower() for c in res_leak["citations"]):
        print("[-] FAILED: Pipeline leaked COA document into DBMS filter scope!")
    else:
        print("[+] PASSED: Zero cross-course leakage. Scoped strictly to DBMS or routed via CRAG.")

if __name__ == "__main__":
    run_tests()