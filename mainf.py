import streamlit as st
import rag_pipelinef as rag  # Import as 'rag' to avoid name collisions
import os
import subprocess
import platform

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

def reveal_in_explorer(file_path):
    # Standardize the path for the current OS
    file_path = os.path.normpath(file_path)
    if platform.system() == "Windows":
        # /select, highlights the specific file in the folder
        subprocess.run(["explorer", "/select,", file_path])
    elif platform.system() == "Darwin": # macOS
        subprocess.run(["open", "-R", file_path])
    else: # Linux
        subprocess.run(["xdg-open", os.path.dirname(file_path)])
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
    st.markdown("### 📄 Document Management. Allow multiple files upload (Max. 15)")  
    uploaded_files = st.file_uploader( "Select one or several files ",
    type=["pdf", "txt", "docx", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"folder_uploader_{st.session_state.upload_key}"
)
    if uploaded_files:
        total = len(uploaded_files)
        if total > 15:
            st.error(f"You've selected {total} files. Max allowed is 15.")
        else:
            for uploaded in uploaded_files:
                filename = uploaded.name
                # 1️⃣ Check if file is already indexed
                if rag.is_file_indexed(filename):
                    st.warning(f"'{filename}' Index Already exists")
                    continue
                # 2️⃣ Save the new file
                path = os.path.join("data", uploaded.name)
                with open(path, "wb") as f:
                    f.write(uploaded.getbuffer())
                # 3️⃣ Indexation
                ##with st.spinner("Indexing document..."):
                ##    rag.ingest_file(path)
                
                with st.spinner("Processing file..."):
                    if uploaded.type.startswith("image"):
                        rag.ingest_image(path)
                    else:
                        rag.ingest_file(path)

            if uploaded.type.startswith("image"): 
                st.image(uploaded, caption=uploaded.name, use_container_width=True)
            
            st.success("Upload & Indexed completed!")
            # Clean file
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
    #st.header("Knowledge Base")
    stats = rag.get_stats()
    #col1, col2 = st.columns(2)
    #col1.metric("Blocks", stats["chunks"], help="Blocks are bite-sized pieces of your documents created so the AI can find and read relevant information faster.")
    #col2.metric("Docs", stats["documents"])
    st.markdown(
        f"""
        <div style="margin-bottom: 10px;">
            <h3 style='margin-bottom: 0px;'>🗄️ Knowledge Base</h3>
            <p style='font-size:14px; color:gray; margin-top: 0px;'>System ingestion statistics</p>
        </div>
        
        <div style="display: flex; justify-content: space-between; background-color: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(151, 151, 151, 0.2);">
            <div style="text-align: center; flex: 1;">
                <span style="color: gray; font-size: 11px; text-transform: uppercase;">Blocks</span><br>
                <span style="font-size: 20px; font-weight: bold;">{stats['chunks']}</span>
            </div>
            <div style="border-left: 1px solid rgba(151, 151, 151, 0.3); margin: 0 15px;"></div>
            <div style="text-align: center; flex: 1;">
                <span style="color: gray; font-size: 11px; text-transform: uppercase;">Docs</span><br>
                <span style="font-size: 20px; font-weight: bold;">{stats['documents']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
# Chat interaction
# -----------------------------
if "messages" not in st.session_state:    # 1️⃣ Initialize messages
    st.session_state.messages = []

for m in st.session_state.messages:       # 2️⃣ Render chat history (user + assistant messages)
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
for key in st.session_state:              # 3️⃣ 🔁 Re-render stored results
    if key.startswith("result_"):
        data = st.session_state[key]

        with st.chat_message("assistant"):
            st.markdown(data["answer"])

            if data["sources"]:
                with st.expander("Sources & Evidence", expanded=True):
                    for item in data["snippets"]:

                        # Preview
                        if item["file"].endswith((".png", ".jpg", ".jpeg")):
                            st.image(f"data/{item['file']}", width=200)
                        else:
                            st.caption(f"Source: {item['file']} | Score: {item['score']}")
                            st.info(item["snippet"])

                        # Button
                        full_path = os.path.abspath(f"data/{item['file']}")

                        if st.button(
                            f"📁 Open Folder",
                            key=f"btn_{key}_{item['file']}"
                        ):
                            reveal_in_explorer(full_path)

                        st.divider()


if prompt := st.chat_input("Ask something about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Create a unique key for this specific interaction's results We use the prompt hash or a counter to keep it unique
        result_key = f"result_{hash(prompt)}"
        if result_key not in st.session_state:
            with st.spinner("Thinking..."):
                answer, sources, snippets = rag.ask(prompt, required_words=keywordsx, mode=match_mode, min_score=min_score)
                # Store everything in session state so it survives the rerun
                st.session_state[result_key] = { "answer": answer, "sources": sources, "snippets": snippets  }
        # 2. Retrieve the data from session state
        data = st.session_state[result_key]
        #st.markdown(answer)
        st.markdown(data["answer"])
        # Show sources + snippets
        if data["sources"]:
             with st.expander("Sources & Evidence"):
                  #for item in snippets:
                  for item in data["snippets"]:
                        # 1. First, show the preview (Image or Text)
                        if item["file"].endswith((".png", ".jpg", ".jpeg")):
                            st.image(f"data/{item['file']}", width=200)
                        else:
                            st.caption(f"Source: {item['file']} | Relevance Score: {item['score']}")
                            st.info(item["snippet"])
                        # 2. Add the "Show in Folder" button right below the preview
                        # We assume your files are in the 'data/' folder relative to your app
                        full_path = os.path.abspath(f"data/{item['file']}") 
                    
                        if st.button(f"📁 Open Folder", key=f"btn_{item['file']}_{item['score']}"):
                            reveal_in_explorer(full_path)
                    
                        st.divider() # Optional: adds a small line between different sources
    st.session_state.messages.append({"role": "assistant", "content": answer})
