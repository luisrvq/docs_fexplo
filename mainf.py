import streamlit as st
import rag_pipelinef as rag  # Import as 'rag' to avoid name collisions
import os

# Used to Clean upload file box
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

# Used to Clean the wipe all data box
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

def clear_all_filters():
    #1 Resetting the UI components
    st.session_state["keyword_filter_key"] = ""
    st.session_state["min_score_key"] = 0.20
    st.session_state["logic_mode_key"] = "Match All"
    # 2. Reset the Chat History in the UI
    st.session_state.messages = [] # Wipes the UI chat hiostory
    # 3. Reset the AI's memory in the backend
    rag.reset_chat_memory()

# -----------------------------
# Sidebar Options
# -----------------------------

st.markdown(
    """
    <h1 style='font-size:32px;'>AI RAG Assistant</h1>
    <p style='font-size:18px; color:gray;'>
        Find insights across your documents using AI
    </p>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### 📄 Document Management")  
    uploaded = st.file_uploader( "Upload document", type=["pdf", "txt", "docx"], key=f"uploader_{st.session_state.upload_key}" )
    if uploaded:
        filename = uploaded.name
        # 1️⃣ Check if file is already indexed
        if rag.is_file_indexed(filename):
            st.warning(f"'{filename}' Index Already exists")
        else:
            # 2️⃣ Save the new file
            path = os.path.join("data", uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            # 3️⃣ Indexation
            with st.spinner("Indexing document..."):
                rag.ingest_file(path)
            st.success("Indexed!")
        st.session_state.upload_key += 1
        st.rerun()
    
    # Get list of files currently in the 'data' folder
    files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]
    if files:
        selected_file = st.selectbox( "Select file to delete", ["-- Select a file --"] + files, index=0 )
        if selected_file != "-- Select a file --":
             if st.button(f"Delete {selected_file}", type="primary"):
                  with st.spinner(f"Deleting {selected_file}..."):
                    rag.delete_single_file(selected_file)
                    st.success(f"Deleted {selected_file}!")
                    st.rerun() # Refresh to update the metrics and file list
    else:
        st.info("No documents indexed.")

    # Show status of documents and chunks
    st.divider()  
    st.header("Knowledge Base")
    stats = rag.get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Chunks", stats["chunks"])
    col2.metric("Docs", stats["documents"])


    ## Delete all the documents from the database
    st.divider()
    st.subheader("⚙️ System Maintenance")
    
    # Safety Checkbox to prevent accidental clicks
    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False
    confirm_reset = st.checkbox( "Confirm Total Reset", value=st.session_state.confirm_reset )
    # Atualiza o estado somente se o usuário clicar
    st.session_state.confirm_reset = confirm_reset
    files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]
    # Desabilita botão se checkbox não marcado ou se não há arquivos
    wipe_disabled = not st.session_state.confirm_reset or len(files) == 0
   
    if st.button("Wipe All Data", type="primary", disabled=wipe_disabled):
        with st.spinner("Deleting all chunks..."):
            rag.reset_full_database()
            st.session_state.messages = [] # Clear chat
            # Resetar checkbox
            st.session_state.confirm_reset = False
            st.success("Database is now empty!")
            st.rerun()

    st.sidebar.markdown( 
        "<h3 style='margin-bottom:5px;'>🔎 Keyword Filter</h3>",
        unsafe_allow_html=True  )
    keywordsx = st.sidebar.text_input( "Keyword Filter", placeholder="e.g. EnterpriseOne, AI...", key="keyword_filter_key", label_visibility="collapsed" )
    match_mode = st.sidebar.radio( "Keyword Logic",
        options=["Match All", "Match Any"], key="logic_mode_key",
        help="'Match All' is stricter. 'Match Any' returns results containing at least one term." )

    st.sidebar.header("🔍 Retrieval Settings")
    min_score = st.sidebar.slider( "Minimum similarity score", min_value=0.0,  max_value=1.0, value=0.2, step=0.01, key="min_score_key" )

    # THE CLEAR BUTTON: It calls our reset function when clicked
    st.button("Clear All Filters", on_click=clear_all_filters, use_container_width=True)
# -----------------------------
# Chat
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask something about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources, snippets = rag.ask(prompt, required_words=keywordsx, mode=match_mode, min_score=min_score)
            st.markdown(answer)
            # Show sources + snippets
            if sources:
                with st.expander("Sources & Evidence"):
                    for item in snippets:
                        st.caption(f"Source: {item['file']} | Relevance Score: {item['score']}")
                        st.info(item["snippet"])
    st.session_state.messages.append({"role": "assistant", "content": answer})
