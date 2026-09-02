from dotenv import load_dotenv
load_dotenv()  
from app.services.rag.retriever import kognit_graph

inputs = {
    "query": "What is Scheduling ?",
    "course_code": "DBMS",
    "top_k": 3,
    "local_chunks": [],
    "final_context": [],
    "crag_decision": "PENDING",
    "answer": "",
    "citations": [],
    "model_used": "",
}

output = kognit_graph.invoke(inputs)

print("--- CRAG DECISION ---")
print(output["crag_decision"])

print("\n--- CITATIONS ---")
for c in output["citations"]:
    print(f"• {c['source']} (Page {c['page']}) [Score: {c['score']}]")

print("\n--- GENERATED ANSWER ---")
print(output["answer"])