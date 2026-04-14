from dotenv import load_dotenv
import os
import chromadb

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "../data"
DB_PATH = "../storage/chroma_db"
COLLECTION_NAME = "documents"

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
query_engine = index.as_query_engine()

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