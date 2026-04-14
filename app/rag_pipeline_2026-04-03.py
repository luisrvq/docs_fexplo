from dotenv import load_dotenv
import os
import chromadb

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "../data"
DB_PATH = "../storage/chroma_db"
COLLECTION_NAME = "documents"

load_dotenv()

# Cheaper embeddings
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# -----------------------------
# Chroma Persistent Client
# -----------------------------
db = chromadb.PersistentClient(path=DB_PATH)

collection = db.get_or_create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection=collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# -----------------------------------------
# Check if DB already populated 2026-04-01
# -----------------------------------------
existing_chunks = collection.count()

if existing_chunks == 0:
    print("No index found. Creating new index...")

    documents = SimpleDirectoryReader(DATA_PATH).load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )

    print("Index created.")
else:
    print(f"Existing index found. Loading... ({existing_chunks} chunks)")

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

print("Chunks in ChromaDB:", collection.count())
collection = index.vector_store._collection
data = collection.get()
sources = set()
for meta in data["metadatas"]:
    if "file_name" in meta:
        sources.add(meta["file_name"])

print("Indexed files:")
for s in sources:
    print("-", s)

# -----------------------------
# Query Engine
# -----------------------------
## query_engine = index.as_query_engine()   2026-04-03
query_engine = index.as_query_engine(similarity_top_k=10)

# -------------------------------
# Hybrid Query Engine  2026-04-03
# -------------------------------
def hybrid_query(question, keyword=None):
    # Step 1: retrieve more candidates
    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(question)

    # Step 2: optional keyword filtering
    if keyword:
        filtered_nodes = []
        for node in nodes:
            text = node.text.lower()
            if keyword.lower() in text:
                filtered_nodes.append(node)
        # fallback if nothing found
        if filtered_nodes:
            nodes = filtered_nodes

    # Step 3: synthesize response
    query_engine = index.as_query_engine(similarity_top_k=10)
    response = query_engine.query(question)

    return str(response)
# -------------------------------
# Hybrid Query Engine  2026-04-03
# -------------------------------


def ask_question(question):
    response = query_engine.query(question)
    return str(response)

# -----------------------------
# Stats (for Streamlit sidebar)
# -----------------------------
def get_stats():
    data = collection.get()

    return {
        "chunks": collection.count(),
        "documents": len(data["documents"])
    }

# ---------------------------------------------------
# Add new files to chroma and index Begin 2026-01-04
# ---------------------------------------------------
#  import os
#  from llama_index.core import SimpleDirectoryReader


from llama_index.core.schema import Document

def ingest_file(uploaded_file):
    DATA_PATH = "../data"
    file_path = os.path.join(DATA_PATH, uploaded_file.name)

    if os.path.exists(file_path): 
        print("File already exists, skipping...", file_path) 
        return

    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load document
    documents = SimpleDirectoryReader(
        input_files=[file_path]
    ).load_data()

    # 🔥 FULL SAFE FLATTEN
    flat_docs = []

    for item in documents:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, Document):
                    flat_docs.append(sub)
        elif isinstance(item, Document):
            flat_docs.append(item)

    # Debug (optional)
    print("Final docs count:", len(flat_docs))
    print("Type check:", type(flat_docs[0]))

    # ✅ FIX: insert one by one
    for doc in flat_docs:
        index.insert(doc)

    print(f"Indexed: {uploaded_file.name}")
# ---------------------------------------------------
# Add new files to chroma and index End    2026-01-04
# ---------------------------------------------------