import streamlit as st
import os
from rag_pipeline import ask_question, get_stats, ingest_file

st.title("AI RAG Assistant")

# Sidebar stats
stats = get_stats()

st.sidebar.header("ChromaDB Stats")
st.sidebar.write(f"Chunks: {stats['chunks']}")
st.sidebar.write(f"Documents: {stats['documents']}")

# -------------------------
# Upload UI
# -------------------------
st.sidebar.header("Upload Documents")

uploaded_file = st.sidebar.file_uploader(
    "Upload file",
    type=["pdf", "txt", "docx"]
)

if uploaded_file is not None:
    with st.spinner("Indexing document..."):
        ingest_file(uploaded_file)

    st.sidebar.success("Document indexed!")
    st.rerun()

# -------------------------
# Chat UI
# -------------------------
question = st.text_input("Ask a question")

if question:
    answer = ask_question(question)
    st.write(answer)
