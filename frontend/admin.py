import streamlit as st
import requests

st.title("KOGNIT Admin — Document Ingestion")

uploaded_file = st.file_uploader("Upload ECS Syllabus or Lecture PDF", type=["pdf"])
course_code = st.selectbox("Select Course", ["COA", "DBMS", "OS", "DSP", "CIRCUITS", "EMT", "IOT"])
unit = st.number_input("Unit / Module", min_value=1, max_value=8, value=1)

if st.button("Upload and Index"):
    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        data = {"course_code": course_code, "unit": unit, "visibility": "global"}
        
        with st.spinner("Parsing, embedding, and indexing into Qdrant..."):
            res = requests.post("http://localhost:8000/documents/upload", files=files, data=data)
            
        if res.status_code == 200:
            st.success(f"Indexed successfully! {res.json()['chunks_indexed']} chunks created.")
        else:
            st.error(f"Error: {res.text}")