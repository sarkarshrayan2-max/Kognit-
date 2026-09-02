import json
import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="KOGNIT", page_icon="🎓", layout="wide")
st.title("🎓 KOGNIT — Academic Assistant")

tab_student, tab_admin = st.tabs(["💬 Student Chat (Streaming)", "📤 Admin: Ingest Document"])

# ==========================================
# 1. STREAMING STUDENT CHAT
# ==========================================
with tab_student:
    st.subheader("Ask KOGNIT")

    col_c, col_k = st.columns([3, 1])
    with col_c:
        selected_course = st.selectbox(
            "Filter by Course",
            ["DBMS", "COA", "OS", "DSP", "CIRCUITS", "EMT", "IOT"],
            key="chat_course_select"
        )
    with col_k:
        top_k = st.slider("Top Chunks", min_value=1, max_value=5, value=3)

    student_query = st.text_input("Your Question:", placeholder="e.g., Explain SQL inner joins")

    if st.button("Submit Question", type="primary"):
        if not student_query.strip():
            st.warning("Please enter a question.")
        else:
            # Reset metadata for the new query
            st.session_state["meta"] = None

            payload = {
                "query": student_query,
                "course_code": selected_course,
                "top_k": top_k,
            }

            try:
                response = requests.post(
                    f"{FASTAPI_URL}/chat/stream",
                    json=payload,
                    stream=True,
                    timeout=180
                )

                if response.status_code != 200:
                    st.error(f"Error {response.status_code}: {response.text}")
                else:
                    st.markdown("### Teacher Explanation")

                    def stream_tokens():
                        try:
                            # Stream lines safely with UTF-8 decoding
                            for line in response.iter_lines(decode_unicode=True):
                                if line:
                                    if line.startswith("data: "):
                                        try:
                                            data = json.loads(line[6:])
                                            msg_type = data.get("type")

                                            if msg_type == "metadata":
                                                st.session_state["meta"] = data
                                            elif msg_type == "token":
                                                yield data.get("content", "")
                                        except json.JSONDecodeError:
                                            continue
                        except requests.exceptions.ChunkedEncodingError:
                            yield "\n\n*(Stream severed prematurely by server)*"

                    # Stream text tokens directly to the UI
                    st.write_stream(stream_tokens)

                    # Display metadata & citations once stream finishes
                    meta = st.session_state.get("meta")
                    if meta:
                        st.divider()
                        crag_decision = meta.get("crag_decision", "UNKNOWN")
                        badge_col = "green" if crag_decision == "CORRECT" else "orange"
                        st.markdown(
                            f"**CRAG Decision:** :{badge_col}[{crag_decision}] | **Model:** `{meta.get('model_used', 'N/A')}`"
                        )

                        citations = meta.get("citations", [])
                        if citations:
                            st.markdown("#### Retrieved Citations")
                            for idx, cite in enumerate(citations, 1):
                                st.markdown(
                                    f"**{idx}. {cite.get('source', 'Unknown')}** (Page {cite.get('page', '?')}) — Confidence: `{cite.get('score', 0.0)}`"
                                )

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI backend. Ensure `uvicorn app.main:app` is running on port 8000.")

# ==========================================
# 2. ADMIN INGESTION TAB
# ==========================================
with tab_admin:
    st.subheader("Upload ECS Course PDF")
    with st.form("ingest_form", clear_on_submit=True):
        course_code = st.selectbox(
            "Target Course",
            ["DBMS", "COA", "OS", "DSP", "CIRCUITS", "EMT", "IOT"],
            key="admin_course_select"
        )
        unit = st.number_input("Unit", min_value=1, max_value=8, value=1)
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

        if st.form_submit_button("Index Document"):
            if not pdf_file:
                st.error("Please upload a PDF file first.")
            else:
                with st.spinner(f"Ingesting and vectorizing '{pdf_file.name}' into Qdrant..."):
                    try:
                        files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                        data = {
                            "course_code": course_code,
                            "unit": unit,
                            "visibility": "global"
                        }
                        res = requests.post(f"{FASTAPI_URL}/documents/upload", files=files, data=data, timeout=180)
                        
                        if res.status_code == 200:
                            result = res.json()
                            st.success(
                                f"Indexed successfully: {result.get('chunks_indexed', 0)} chunks created for {result.get('course_code')} (Unit {result.get('unit')})."
                            )
                        else:
                            st.error(f"Error {res.status_code}: {res.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to FastAPI backend. Ensure `uvicorn app.main:app` is running on port 8000.")