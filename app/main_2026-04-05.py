import streamlit as st
import os
from rag_pipeline import ask_question, get_stats, ingest_file, hybrid_query

st.title("AI RAG Assistant")

# Sidebar stats
stats = get_stats()

st.sidebar.header("ChromaDB Stats")
st.sidebar.write(f"Chunks: {stats['chunks']}")
st.sidebar.write(f"Documents: {stats['documents']}")

# -------------------------
# Upload UI
# -------------------------
keywordx = st.text_input("Optional keyword filter")  # 2026-04-03


st.sidebar.header("Upload Documents")

# Initialize session state  2026-04-02
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


uploaded_file = st.sidebar.file_uploader(
    "Upload file",
    type=["pdf", "txt", "docx"]
)

if uploaded_file is not None:

    # ✅ Only process if it's a NEW file
    if st.session_state.processed_file != uploaded_file.name:

        with st.spinner("Indexing document..."): 
              ingest_file(uploaded_file)

        st.sidebar.success("Document indexed!")
        st.session_state.processed_file = uploaded_file.name


# -------------------------
# Chat UI
# -------------------------
question = st.text_input("Ask a question")

if question:
##    answer = ask_question(question)
    answer, sources = hybrid_query(question, keyword=keywordx)
    st.write(answer)
    st.markdown("### 📂 Sources")
    for s in sources:
         st.write(f"- {s}")
