from dotenv import load_dotenv
import chromadb

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# Load documents
documents = SimpleDirectoryReader("../data").load_data()

# Embedding model (cheap + fast)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Persistent DB
db = chromadb.PersistentClient(path="storage/chroma_db")
collection = db.get_or_create_collection("documents")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index ONCE
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)

print("✅ Index built and stored in ChromaDB")