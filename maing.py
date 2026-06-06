# maing.py
#---------
import streamlit as st
import rag_pipelineg as rag  # Import as 'rag' to avoid name collisions
import os
import subprocess
import platform
import uuid     # Get user session_id for unique file keys
from agent_manager import AssistantAgent  # Import the Agent manager class

# ----------------------------------------------
# First visit → create session_id → store in URL  
# Refresh → read from URL → reuse 
# ----------------------------------------------
query_params = st.query_params
if "session_id" in query_params:
    session_id = query_params["session_id"]
    st.session_state.session_id = session_id
    print(f"Existing session_id in URL: {session_id}")
else:
    session_id = str(uuid.uuid4())[:8]
    st.session_state.session_id = session_id
    st.query_params["session_id"] = session_id

user_data_path = f"data/{st.session_state.session_id}"
os.makedirs(user_data_path, exist_ok=True)

# Used to Clean upload file box 
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

# 1. First, keep your RAG initialization as is
if "rag" not in st.session_state:
    st.session_state.rag = rag.MyRagSystem(st.session_state.session_id)
rag_system = st.session_state.rag
print(f"rag_system initialized for session_id: {st.session_state.session_id}")

# 2. NEW: Initialize the Agent Manager right after the RAG system
if "agent" not in st.session_state:
    st.session_state.agent = AssistantAgent(rag_system)  # We pass the existing rag_system instance to the agent
assistant_agent = st.session_state.agent
print(f"assistant_agent initialized for session_id: {st.session_state.session_id}")

# 3. Keep your FAQ logic (it still uses the rag_system directly for ingestion)
# Get the directory where your script is running
if "faqs_loaded" not in st.session_state:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    faq_path = os.path.join(base_dir, "faqs_maing.json")
    faq_docs = rag_system.load_faqs(faq_path) # Load FAQs from a JSON file
    rag_system.index.insert_nodes(faq_docs)   # Add FAQs to your current index alongside your other documents
    st.session_state.faqs_loaded = True

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

def initialize_session(session_id):
    st.session_state.session_id = session_id
    st.session_state.rag = rag.MyRagSystem(session_id)
    st.session_state.agent = AssistantAgent( st.session_state.rag  )   # Re-initialize the agent with the new RAG system
    st.session_state.messages = []

# -----------------------------
# Title of Chat pannel
# -----------------------------
st.markdown(
    """
    <h1 style='font-size:32px;'>AI Personal Assistant</h1>
    <p style='font-size:18px; color:gray;'>
        Find information across your documents using AI 
    </p>
    """,
    unsafe_allow_html=True ) 

# -----------------------------
# Sidebar Options
# -----------------------------
with st.sidebar:
    # -----------------------------
    # SESSION TOOLBAR
    # -----------------------------
    st.header("🔍 Session options")
    ##session_id = st.session_state.session_id
    session_col1, session_col2 = st.columns([3, 1])
    with session_col1:
        st.markdown( f""" <div style="
                            background-color: rgba(255,255,255,0.04);
                            border: 1px solid rgba(151,151,151,0.2);
                            border-radius: 8px;
                            padding: 10px; text-align: center; ">
                            <div style="
                                font-size:10px; color:gray; text-transform:uppercase; margin-bottom:2px; ">
                                Active Session </div>
                            <div style=" font-size:18px; font-weight:bold; color:#55aaff; ">  {session_id} </div>
                            </div> """,
                            unsafe_allow_html=True  )

    with session_col2:
        st.markdown( """ <div style="
                text-align:center;
                font-size:12px;
                color:gray;
                margin-bottom:4px; "> New Session </div> """,
            unsafe_allow_html=True
    )
        if st.button( "➕", help="Start New Session", key="new_session_btn", use_container_width=True ):
            new_id = str(uuid.uuid4())[:8]
            st.session_state.clear()
            st.query_params["session_id"] = new_id
            st.session_state.session_id = new_id
            st.session_state.messages = []
            rag_system.reset_memory()
            st.rerun()

    ## Open Existing Session
    st.sidebar.markdown("### 🔄 Open existing Session")
    # Create 3 columns: input | open | cancel Using columns to align the input and buttons perfectly
    col_input, col_go, col_x = st.columns([3, 1, 1])
    with col_input:
        new_session_id = st.text_input(
            "Enter session ID", placeholder="e.g. ab12cd34",  label_visibility="collapsed" ) # 🔥 removes extra label spac
    with col_go:
        st.markdown('<div style="margin-top: 2px;">', unsafe_allow_html=True)  # Using a slight top margin to align with the text input
        open_clicked = st.button("➡️", help="Open session", key="switch_go")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_x:
        st.markdown('<div style="margin-top: 2px;">', unsafe_allow_html=True)
        cancel_clicked = st.button("❌", help="Clear", key="switch_clear")
        st.markdown('</div>', unsafe_allow_html=True)
    # Handle actions
    if open_clicked:
        if new_session_id:
            st.session_state.switch_session_input = ""
            st.query_params["session_id"] = new_session_id
            initialize_session(new_session_id)  # converted as a 
            st.session_state.rag.reset_memory()
            st.session_state.rag.reload_spreadsheet_store()  # NEW: Reload spreadsheet data for the new session
            st.rerun()
        else:
            st.warning("Enter a session ID")

    if cancel_clicked:
        st.session_state.switch_session_input = ""
        st.rerun()


    ## Start fresh seartch
    st.checkbox( "Fresh search (ignore previous answers)", key="fresh_search", value=True )
    # Atualiza o estado somente se o usuário clicar
    fresh_search = st.session_state.fresh_search


    ## DOCUMENT MANAGEMENT SECTION
    st.divider()
    st.markdown("### 📄 Document Management") 
    ## Upload files section 
    uploaded_files = st.file_uploader( "Select one or several files (Max. 15) ",
    type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "csv", "xlsx"],
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
            st.error(f"You've selected {total} files. Max allowed is 15. Delete the excess files and will process remaining files automatically ")
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
                    elif uploaded.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                                           "application/vnd.ms-excel" ]:  
                        # Excel ingestion (multiple sheets) 
                        # 1) Store DataFrame for spreadsheet_search tool
                        rag_system.load_spreadsheet_store(path, filename)
                        # 2) Index sheets as markdown tables for vector search
                        excel_docs = rag_system.load_excel_all_sheets(path, filename) 
                        for doc in excel_docs:
                            rag_system.index.insert(doc) # Directly insert each sheet
                    else:
                        rag_system.ingest_file(path)

            st.session_state.files_processed = True
            st.success("Upload & Indexed completed!")
            # Reset uploader
            st.session_state.upload_key += 1
            st.rerun()

    ## Show statistics of # of documents and chunks
    stats = rag_system.get_stats()
    st.markdown("#### 🗄 Knowledge Base <span style='font-size: 14px; color: gray;'>(ingestion statistics)</span>", unsafe_allow_html=True)
    # Create two columns for the side-by-side look
    col_blocks, col_docs, col_faq = st.columns(3)

    with col_blocks:
        st.markdown(
            f"""
            <div style="background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(151,151,151,0.2); 
                        border-radius: 8px; padding: 10px; text-align: center;">
                <p style="color: gray; font-size: 10px; margin: 0; text-transform: uppercase;">Blocks</p>
                <p style="font-size: 18px; font-weight: bold; margin: 0;">{stats['chunks']}</p>
            </div>
            """, unsafe_allow_html=True   )

    with col_docs:
        st.markdown(
            f"""
            <div style="background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(151,151,151,0.2); 
                        border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 15px;">   <!-- Added margin here -->
                <p style="color: gray; font-size: 10px; margin: 0; text-transform: uppercase;">Docs</p>
                <p style="font-size: 18px; font-weight: bold; margin: 0;">{stats['documents']}</p>
            </div>
            """, unsafe_allow_html=True )
        
    with col_faq:
        st.markdown(
            f"""
            <div style="background-color: rgba(255, 255, 255, 0.05);
                        border: 1px solid rgba(151,151,151,0.2);
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                        margin-bottom: 15px;">
                <p style="color: gray; font-size: 10px; margin: 0;
                        text-transform: uppercase;">FAQs</p>
                <p style="font-size: 18px; font-weight: bold; margin: 0;">
                    {stats['faq_documents']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ------------------------------------
    # 👁️ View / 🗑 Delete Documents (NEW)
    # ------------------------------------
    with st.expander("👁️ Documents", expanded=False):
        files = sorted([ f for f in os.listdir(user_data_path) if os.path.isfile(os.path.join(user_data_path, f)) ])
        if not files:
            st.info("No documents uploaded.")
        else:
            # Init confirmation state
            if "confirm_delete_file" not in st.session_state:
                st.session_state.confirm_delete_file = None
            for i, file in enumerate(files):
                file_path = os.path.join(user_data_path, file)
                # -----------------------------
                # File Icon
                # -----------------------------
                if file.endswith(".pdf"):
                    iconx = "📕"
                elif file.endswith(".docx"):
                    iconx = "📘"
                elif file.endswith(".xlsx"):
                    iconx = "📊"
                elif file.endswith((".png", ".jpg", ".jpeg")):
                    iconx = "🖼️"
                else:
                    iconx = "📄"
                short_name = file if len(file) < 35 else file[:32] + "..."
                # -----------------------------
                # Layout
                # -----------------------------
                col1, col2, col3 = st.columns([7, 0.6, 1.4])
                # File name
                with col1:
                    st.markdown( f"""
                        <div style=" padding-top: 8px; padding-bottom: 4px; font-size: 15px; overflow-wrap: anywhere; ">
                            {iconx} <b>{short_name}</b>
                        </div> """,
                        unsafe_allow_html=True )
                # Delete button
                with col2:
                    disabled = ( st.session_state.confirm_delete_file is not None and st.session_state.confirm_delete_file != file )
                    st.markdown(""" <style>
                        div[data-testid="stButton"] button[kind="secondary"] 
                            { padding: 0.15rem 0.35rem;
                            min-height: 28px;
                            font-size: 12px; }
                        </style> """,
                        unsafe_allow_html=True )  
                    if st.button( "🗑", key=f"delete_{i}", disabled=disabled, help=f"Delete {file}" ):
                        st.session_state.confirm_delete_file = file
                # Confirmation buttons
                with col3:
                    st.markdown(""" <style>
                                div[data-testid="stButton"] button  padding: 0.12rem 0.30rem; min-height: 26px; font-size: 11px;
                                    border-radius: 6px; }
                                </style> """, 
                                unsafe_allow_html=True)                    
                    if st.session_state.confirm_delete_file == file:
                        confirm_col, cancel_col = st.columns([1, 1])
                        with confirm_col:
                            if st.button("✔", key=f"confirm_{i}", help="Confirm delete"):
                                rag_system.delete_single_file(file)
                                st.toast( f"{file} deleted", icon="🗑" )
                                st.session_state.confirm_delete_file = None
                                st.rerun()
                        with cancel_col:
                            if st.button("✖", key=f"cancel_{i}", help="Cancel delete"):
                                st.session_state.confirm_delete_file = None
                                st.rerun()
    ## ----------------------   

    # -----------------------------    
    # Filters
    # -----------------------------
    st.sidebar.markdown( 
        "<h3 style='margin-bottom:5px;'>🔎 Keyword Filter</h3>",
        unsafe_allow_html=True  )
    keywordsx = st.sidebar.text_input( "Keyword Filter", placeholder="e.g. Oracle, SAP, AI...", key="keyword_filter_key", label_visibility="collapsed" )
    match_mode = st.sidebar.radio( "Keyword Logic",
        options=["Match All", "Match Any"], key="logic_mode_key",
        help="'Match All' is stricter. 'Match Any' returns results containing at least one term." )
    
    ## Retrieval settings
    st.sidebar.header("🔍 Retrieval Settings")
    min_score = st.sidebar.slider( "Minimum similarity score", min_value=0.0,  max_value=1.0, value=0.20, step=0.01, key="min_score_key" )

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
                        if item.get("type") == "official_faq":
                            st.caption( f"❓ FAQ Knowledge | "
                                        f"Similarity: {item.get('similarity_score', 0):.2f} | "
                                        f"Relevance: {item.get('relevance_score', 0):.2f}"   )
                        else:
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
                        if item.get("type") == "official_faq":
                            st.caption( f"❓ FAQ Knowledge | "
                                        f"Similarity: {item.get('similarity_score', 0):.2f} | "
                                        f"Relevance: {item.get('relevance_score', 0):.2f}"   )
                        else:
                            st.caption( f"📄 {item['file']} | "
                                        f"Similarity: {item.get('similarity_score', 0):.2f} | "
                                        f"Relevance: {item.get('relevance_score', 0):.2f}" )
                            st.info(item["snippet"])

                        # Button
                        full_path = os.path.abspath(f"{user_data_path}/{item['file']}")

                        render_file_action( file_name=item["file"], full_path=full_path, key=f"{i}_{j}" )

                        st.divider()


if prompt := st.chat_input("Ask something about your uploaded documents or FAQs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # NEW: use the Agent manager to handle the query and get the response + sources
            # This method will internally call the RAG system, apply any necessary logic, 
            # and return the final answer along with the sources and snippets.
            answer, sources, snippets = assistant_agent.handle_query(prompt) 
            print(f"After handle_query - Answer: {answer}")
            print(f"After handle_query - Sources: {sources}")
            
            if isinstance(answer, list):    # If spreadsheet results (list of dicts)
                for r in answer: 
                    st.markdown(f"### 📄 {r['file']}")
                    st.dataframe(r["rows"])
            else:
                st.markdown(answer)
            ##st.markdown(answer)
            
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
## --------------------------- 00 --------------------------- ##