from dotenv import load_dotenv
import chromadb

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

def get_query_engine():
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    db = chromadb.PersistentClient(path="storage/chroma_db")
    collection = db.get_or_create_collection("documents")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    return index.as_query_engine()