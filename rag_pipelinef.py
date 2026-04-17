import os
from urllib import response
from dotenv import load_dotenv
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings, )

from llama_index.core import query_engine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.core.chat_engine import ContextChatEngine  # Needed for Chat History

from llama_index.core.schema import QueryBundle
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import KeywordNodePostprocessor
from llama_index.readers.file import DocxReader

load_dotenv()

class ManualNodeRetriever(BaseRetriever):
    def __init__(self, nodes):
        self._nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle):
        return self._nodes

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data"
DB_PATH = "storage/chroma_db_fresh"
COLLECTION_NAME = "documents_fresh"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# -----------------------------
# Chroma setup
# -----------------------------
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# -----------------------------
# Build or load index
# -----------------------------
# Check what's actually in Chroma right now
existing_chunks = collection.count()
# Check if the data directory exists and has files
# Check if there are any physical files in the data folder
##has_physical_files = any(os.scandir(DATA_PATH))
data_exists = os.path.exists(DATA_PATH) and any(os.scandir(DATA_PATH))

if existing_chunks == 0:
    if data_exists:
        loader = SimpleDirectoryReader(input_dir=DATA_PATH)
        docs = loader.load_data()
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True, )
    else:
        # Create an empty index if no files are found
        # This allows the app to start so you can use the Upload button
        index = VectorStoreIndex.from_documents( [], storage_context=storage_context   )
else:
    index = VectorStoreIndex.from_vector_store(vector_store)

# -----------------------------
# Memory
# -----------------------------
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Persist memory during the session
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
# -----------------------------
# Query function ASK function
# -----------------------------
def ask(question: str, required_words: str = "", mode: str = "Match All", min_score: float = 0.15):
    # 1. Retrieve a generous pool
    retriever = index.as_retriever(similarity_top_k=20)
    initial_nodes = retriever.retrieve(question)

    # 2. Manual Keyword Filtering Logic
    if required_words:
        keywords_split  = [k.strip().lower() for k in required_words.split(",") if k.strip()]
        filtered_nodes = []
        for node_with_score in initial_nodes:
            content = node_with_score.node.get_content().lower()
            
            if mode == "Match All":
                # Only keep if ALL keywords are present
                if all(kw in content for kw in keywords_split):
                    filtered_nodes.append(node_with_score)
            else:   # Match Any
                # Keep if ANY keyword is present
                if any(kw in content for kw in keywords_split):
                    filtered_nodes.append(node_with_score)
        
        nodes_to_rerank = filtered_nodes
    else:    
        nodes_to_rerank = initial_nodes

    # SAFETY CHECK: If no nodes survived filtering, return early
    if not nodes_to_rerank:
        return "I found documents matching your vector search, but none contained the required keywords. Try changing 'Match All' to 'Match Any'.", [], []

    # 3. Rerank the remaining nodes
    llm = OpenAI(model="gpt-4o-mini")
    reranker = RankGPTRerank(llm=llm, top_n=3)
    reranked_nodes = reranker.postprocess_nodes(nodes_to_rerank, query_bundle=QueryBundle(question))

    # 4. Generate the final answer using the reranked nodes
    # This retriever ONLY returns the nodes you already filtered and ranked
    custom_retriever = ManualNodeRetriever(reranked_nodes)

    chat_engine = ContextChatEngine.from_defaults(
        retriever=custom_retriever, # This is required by the constructor
        memory=memory,       # Uses your 'memory' variable defined at the top of your file
        system_prompt="You are a helpful assistant. Use the provided context to answer questions."
    )

    # 5. Execute the chat normally (no 'nodes' argument needed now)
    response = chat_engine.chat(question)
    final_source_nodes = response.source_nodes 
    
    # 6. Build sources + snippets
    ranked_sources = []
    snippets = []
    seen = set()

    ##for node_with_score in reranked_nodes:
    for i, node_with_score in enumerate(reranked_nodes):
        score = 1.0 - (i * 0.1)

        node = node_with_score.node
        fname = node.metadata.get("file_name", "Unknown")

        if fname not in seen:
            ranked_sources.append((fname, round(score, 3)))
            seen.add(fname)

        snippet = node.get_content()[:250].replace("\n", " ").strip()
        snippets.append({"file": fname, "snippet": snippet, "score": round(score, 3)})

    return response.response, ranked_sources, snippets


# -----------------------------
# Ingestion
# -----------------------------
def ingest_file(path: str):
    # This explicitly tells LlamaIndex how to handle Word files
    docx_parser = DocxReader()
    file_extractor = {".docx": docx_parser}
    #loader = SimpleDirectoryReader(input_files=[path])
    loader = SimpleDirectoryReader( input_files=[path], file_extractor=file_extractor ) 
    docs = loader.load_data()
    for d in docs:
        index.insert(d)

# -----------------------------
# Stats
# -----------------------------
def get_stats():
    metas = collection.get().get("metadatas", [])
    unique_docs = len(set(m.get("file_name") for m in metas if m)) if metas else 0
    return {"chunks": collection.count(), "documents": unique_docs}
# ----------------------------------
# Reset chat memory to its defaults
# ----------------------------------
def reset_chat_memory():
    # Make sure 'memory' matches the variable name 
    # you defined at the top of this file
    memory.reset() 

# ----------------------------------
# Persist function
# ----------------------------------
def force_persist():
    try:
        collection.persist()
    except:
        pass

# ----------------------------------
# ChromaDB Single deletion file
# ----------------------------------
def delete_single_file(filename: str):
    # 1. Delete from ChromaDB using a metadata filter
    # This removes all chunks belonging to that specific file
    collection.delete(where={"file_name": filename})
    
    # 2. Delete the physical file from your data folder
    file_path = os.path.join(DATA_PATH, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # 3. Refresh the index to reflect the changes
    global index
    index = VectorStoreIndex.from_vector_store(vector_store)

def reset_full_database():
    # 1. Delete and recreate the collection in Chroma
    client.delete_collection(COLLECTION_NAME)
    
    # 2. Re-initialize the globals
    global collection, vector_store, storage_context, index
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 3. Create a clean empty index
    index = VectorStoreIndex.from_documents([], storage_context=storage_context)
    
    # 4. Clean local data folder
    for f in os.listdir(DATA_PATH):
        os.remove(os.path.join(DATA_PATH, f))

# Check if we already have these files to avoid duplicates
def is_file_indexed(filename):
    results = collection.get(where={"file_name": filename})
    return len(results['ids']) > 0