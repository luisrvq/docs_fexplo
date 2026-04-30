import streamlit as st
import rag_pipelineg as rag  # Import as 'rag' to avoid name collisions
import os
import subprocess
import platform
import uuid     # Get user session_id for unique file keys

# ----------------------------------------------
# First visit → create session_id → store in URL  
# Refresh → read from URL → reuse 
# ----------------------------------------------

query_params = st.query_params
if "session_id" in query_params:
    session_id = query_params["session_id"]
    st.session_state.session_id = session_id
else:
    session_id = str(uuid.uuid4())[:8]
    st.session_state.session_id = session_id
    st.query_params["session_id"] = session_id

user_data_path = f"data/{st.session_state.session_id}"
os.makedirs(user_data_path, exist_ok=True)

# Used to Clean upload file box  (double check if needed)
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

# Used to Clean the wipe all data box
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

# ----------------------------------------------
# Initialize backend for this session
# ----------------------------------------------
data_path = f"data/{st.session_state.session_id}"
db_path = f"storage/{st.session_state.session_id}"
collection_name = f"documents_{st.session_state.session_id}"

if "rag" not in st.session_state:
    st.session_state.rag = rag.MyRagSystem(st.session_state.session_id)
rag_system = st.session_state.rag

# ----------------------------------------------
# UI Helpers
# ----------------------------------------------
# Check if the app is running on the web
def render_file_action(file_name: str, full_path: str, key: str):
    if not os.path.exists("C:/"):
        # 🌐 Web behavior
        st.caption(f"📄 {file_name}")
        try:
            with open(full_path, "rb") as f:
                st.download_button( label="⬇️ Download", data=f, file_name=file_name, key=f"dl_{key}" )
        except:
            st.caption("File not available")
    else:
        # 💻 Local behavior
        if st.button("📁 Open Folder", key=f"btn_{key}"):
            reveal_in_explorer(full_path)

def clear_all_filters():
    #1 Resetting the UI components
    st.session_state["keyword_filter_key"] = ""
    st.session_state["min_score_key"] = 0.20
    st.session_state["logic_mode_key"] = "Match All"
    # 2. Reset the Chat History in the UI
    st.session_state.messages = [] # Wipes the UI chat hiostory
    # 3. Reset the AI's memory in the backend
    rag_system.reset_memory()

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
# Title of Chat pannel
# -----------------------------
st.markdown(
    """
    <h1 style='font-size:32px;'>AI RAG Personal Assistant</h1>
    <p style='font-size:18px; color:gray;'>
        Find information across your documents using AI 
    </p>
    """,
    unsafe_allow_html=True ) 

fresh_search = st.checkbox( "🔄 Fresh search (ignore previous answers)", 
                           value=True ) # ✅ default ON 
# -----------------------------
# Sidebar Options
# -----------------------------
with st.sidebar:
    st.header("🔍 Session options")
    session_id = st.session_state.session_id
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; background-color: rgba(255, 255, 255, 0.05); 
            padding: 10px; border-radius: 8px; border: 1px solid rgba(151, 151, 151, 0.2);">
            <div style="text-align: left; flex: 1;">
                <span style="font-size: 18px; font-weight: bold;">Current session_id: {session_id} </span>
            </div>            
        """,
        unsafe_allow_html=True  ) 

    ## Switch session
    st.sidebar.markdown("### 🔄 Switch Session")
    # Create 3 columns: input | open | cancel
    col1, col2, col3 = st.sidebar.columns([4, 1, 1])
    with col1:
        new_session_id = st.text_input(
            "Enter session ID",
            placeholder="e.g. ab12cd34",
            label_visibility="collapsed" ) # 🔥 removes extra label spac
    with col2:
        open_clicked = st.button("➡️", help="Open session")
    with col3:
        cancel_clicked = st.button("❌", help="Cancel")
    # Handle actions
    if open_clicked:
        if new_session_id:
            st.query_params["session_id"] = new_session_id
            st.session_state.session_id = new_session_id
            st.session_state.switch_session_input = ""
            st.session_state.messages = [] # Wipes the UI chat hiostory
            rag_system.reset_memory()
            st.session_state.rag = rag.MyRagSystem(new_session_id)
            ##rag_system.initialize_paths( f"data/{new_session_id}", f"storage/{new_session_id}", f"documents_{new_session_id}" ) # Re-init backend with new paths
            st.rerun()
        else:
            st.warning("Enter a session ID")

    if cancel_clicked:
        st.session_state.switch_session_input = ""
        st.rerun()

    ## Reset session buttonn
    #st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start New Session"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.clear()   # Optional: clear chat memory
        st.query_params["session_id"] = new_id
        st.session_state.session_id = new_id
        st.session_state.messages = [] # Wipes the UI chat hiostory
        rag_system.reset_memory()
        ##rag_system.initialize_paths( f"data/{new_id}", f"storage/{new_id}", f"documents_{new_id}" ) # Re-init backend with new paths
        st.rerun()

    ## Upload files section
    st.divider()
    st.markdown("### 📄 Document Management")  
    uploaded_files = st.file_uploader( "Select one or several files (Max. 15) ",
    type=["pdf", "txt", "docx", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"folder_uploader_{st.session_state.upload_key}" )
    if "files_processed" not in st.session_state:  #Initialize flag
        st.session_state.files_processed = False
    # ✅ RESET flag when user selects new files
    if uploaded_files:
        st.session_state.files_processed = False
    # Main process
    if uploaded_files and not st.session_state.files_processed:
        total = len(uploaded_files)
        if total > 15:
            st.error(f"You've selected {total} files. Max allowed is 15.")
        else:
            for uploaded in uploaded_files:
                filename = uploaded.name
                # 1️⃣ Check if file is already indexed
                if rag_system.is_file_indexed(filename):
                    st.warning(f"'{filename}' Index Already exists")
                    continue
                # 2️⃣ Save the new file
                path = os.path.join(user_data_path, uploaded.name)
                with open(path, "wb") as f:
                    f.write(uploaded.getbuffer())
                # 3️⃣ Process file
                with st.spinner(f"Processing {filename}..."):
                    if uploaded.type.startswith("image"):
                        rag_system.ingest_image(path)
                        st.image(uploaded, caption=filename, use_container_width=True)
                    else:
                        rag_system.ingest_file(path)

            st.session_state.files_processed = True
            st.success("Upload & Indexed completed!")
            # Reset uploader
            st.session_state.upload_key += 1
            st.rerun()
    ## ----------------------
    ## Delete files section  
    ## ----------------------  
    # Get list of files currently in the 'data' folder
    if "show_delete_panel" not in st.session_state:  # Init state
        st.session_state.show_delete_panel = False
    
    if "pending_delete" not in st.session_state:     # Init state
        st.session_state.pending_delete = None
    
    # 🔘 Show ONLY when panel is closed
    if not st.session_state.show_delete_panel:
        if st.button("🗑️ Delete a file", key="open_delete_panel"):
            st.session_state.show_delete_panel = True

    # 📂 Delete panel
    if st.session_state.show_delete_panel:
         # 🔝 Top close button
        if st.button("❌ Close Delete Panel", key="close_top"):
            st.session_state.show_delete_panel = False
            st.rerun()

        st.markdown("### 📂 Select a file to delete") 
        files = [f for f in os.listdir(user_data_path) if os.path.isfile(os.path.join(user_data_path, f))]
        if files:
            for i, file in enumerate(files):
                file_path = os.path.join(user_data_path, file)
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(f"📄 **{file}**")
                with col2:
                    if st.button("🗑️", key=f"delete_{i}", help='Delete file'):
                        st.session_state["pending_delete"] = file
        else:
            st.info("No documents available.")
    
        if st.button("❌ Close Delete Panel", key="close_bottom"):  
            st.session_state.show_delete_panel = False
            st.rerun()
    
    # ⚠️ Confirmation
    if st.session_state.pending_delete:
        file = st.session_state.pending_delete
        st.warning(f"⚠️ Delete **{file}**?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm", key="confirm_delete"):
                rag_system.delete_single_file(file)
                st.success("Deleted!")
                st.session_state.pending_delete = None
                st.rerun()
        with col2:
            if st.button("❌ Cancel", key="cancel_delete"):
                st.session_state.pending_delete = None   
    ## ----------------------   
    
    ## Show statistics of # of documents and chunks
    st.divider()  
    stats = rag_system.get_stats()
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
        unsafe_allow_html=True  )

    ## Filters
    st.sidebar.markdown( 
        "<h3 style='margin-bottom:5px;'>🔎 Keyword Filter</h3>",
        unsafe_allow_html=True  )
    keywordsx = st.sidebar.text_input( "Keyword Filter", placeholder="e.g. Oracle, SAP, AI...", key="keyword_filter_key", label_visibility="collapsed" )
    match_mode = st.sidebar.radio( "Keyword Logic",
        options=["Match All", "Match Any"], key="logic_mode_key",
        help="'Match All' is stricter. 'Match Any' returns results containing at least one term." )
    
    ## Retrieval settings
    st.sidebar.header("🔍 Retrieval Settings")
    min_score = st.sidebar.slider( "Minimum similarity score", min_value=0.0,  max_value=1.0, value=0.2, step=0.01, key="min_score_key" )

    # THE CLEAR BUTTON: It calls our reset function when clicked
    st.button("Clear All Filters", on_click=clear_all_filters, use_container_width=True)

    ## Delete all the documents from the database
    st.divider()
    st.subheader("⚙️ System Maintenance")
    
    # Safety Checkbox to prevent accidental clicks
    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False
    confirm_reset = st.checkbox( "Confirm Total Reset", value=st.session_state.confirm_reset )
    # Atualiza o estado somente se o usuário clicar
    st.session_state.confirm_reset = confirm_reset
    files = [f for f in os.listdir(user_data_path) if os.path.isfile(os.path.join(user_data_path, f))]
    # Desabilita botão se checkbox não marcado ou se não há arquivos
    wipe_disabled = not st.session_state.confirm_reset or len(files) == 0
   
    if st.button("🧹 Wipe Current Session Data", type="primary", disabled=wipe_disabled):
        with st.spinner("Deleting all chunks..."):
            rag_system.reset_full_database()
            st.session_state.messages = [] # Clear chat
            # Resetar checkbox
            st.session_state.confirm_reset = False
            st.success("Database is now empty!")
            st.rerun()
    
# -----------------------------
# Chat messages
# -----------------------------
if "messages" not in st.session_state:    # 1️⃣ Initialize messages
    st.session_state.messages = []

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        # Only assistant messages have sources
        if m["role"] == "assistant" and "sources" in m:

            with st.expander("Sources & Evidence", expanded=False):
                for j, item in enumerate(m["snippets"]):

                    # Preview
                    if item["file"].endswith((".png", ".jpg", ".jpeg")):
                        st.image(f"{user_data_path}/{item['file']}", width=200)
                    else:
                        ##st.caption(f"Source: {item['file']} | Score: {item['score']}")
                        st.caption( f"📄 {item['file']} | "
                                    f"Similarity: {item.get('similarity_score', 0):.2f} | "
                                    f"Relevance: {item.get('relevance_score', 0):.2f}" )
                        st.info(item["snippet"])

                    full_path = os.path.abspath(f"{user_data_path}/{item['file']}")

                    # Unique key per message + item
                    render_file_action( file_name=item["file"], full_path=full_path, key=f"{i}_{j}" )

                    st.divider()
for key in st.session_state:              # 3️⃣ 🔁 Re-render stored results
    if key.startswith("result_"):
        data = st.session_state[key]

        with st.chat_message("assistant"):
            st.markdown(data["answer"])

            if data["sources"]:
                with st.expander("Sources & Evidence", expanded=True):
                    for i, item in enumerate (data["snippets"]):

                        # Preview
                        if item["file"].endswith((".png", ".jpg", ".jpeg")):
                            st.image(f"{user_data_path}/{item['file']}", width=200)
                        else:
                            ##st.caption(f"Source: {item['file']} | Score: {item['score']}")
                            st.caption( f"📄 {item['file']} | "
                                        f"Similarity: {item.get('similarity_score', 0):.2f} | "
                                        f"Relevance: {item.get('relevance_score', 0):.2f}"  )
                            st.info(item["snippet"])

                        # Button
                        full_path = os.path.abspath(f"{user_data_path}/{item['file']}")

                        render_file_action( file_name=item["file"], full_path=full_path, key=f"{i}_{j}" )

                        st.divider()


if prompt := st.chat_input("Ask something about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # NEW: build context-aware prompt
            #if "last_source" in st.session_state:
            #    enhanced_prompt = f"""
            #        Focus ONLY on the document: {st.session_state['last_source']}

            #        User question:
            #        {prompt}
            #        """
            #else:
            #    enhanced_prompt = prompt

            answer, sources, snippets = rag_system.ask( prompt, required_words=keywordsx, 
                                                       mode=match_mode, min_score=min_score, reset_memory = fresh_search )

            if sources:
                st.session_state["last_source"] = sources[0][0]  # NEW: store top document

            st.markdown(answer)
        # 2. Retrieve the data from session state
        # Show sources + snippets
        if sources:
             with st.expander("Sources & Evidence"):
                  for j, item in enumerate(snippets):
                        # 1. First, show the preview (Image or Text)
                        if item["file"].endswith((".png", ".jpg", ".jpeg")):
                            st.image(f"{user_data_path}/{item['file']}", width=200)
                        else:
                            st.caption( f"📄 {item['file']} | " 
                                        f"Similarity: {item.get('similarity_score', 0):.2f} | "
                                        f"Relevance: {item.get('relevance_score', 0):.2f}" )
                            st.info(item["snippet"])
                        # 2. Add the "Show in Folder" button right below the preview
                        full_path = os.path.abspath(f"{user_data_path}/{item['file']}") 
                    
                        render_file_action( file_name=item["file"], full_path=full_path, key=f"new_{j}" )
                    
                        st.divider() # Optional: adds a small line between different sources
    st.session_state.messages.append({ "role": "assistant", "content": answer, "sources": sources, "snippets": snippets })