import streamlit as st
from rag_engine import get_query_engine
from rag_pipeline import get_stats # code added 2026-04-01

st.title("AI Document Assistant")

# Show Chroma stats -Start 2026-04-01
stats = get_stats()
st.sidebar.header("ChromaDB Stats")
st.sidebar.write(f"Chunks: {stats['chunks']}")
st.sidebar.write(f"Documents: {stats['documents']}")
# Show Chroma stats - End

query_engine = get_query_engine()

question = st.text_input("Ask a question about your documents")

if question:
    response = query_engine.query(question)
    st.write(response)