import os
import shutil
import base64
from unittest import loader
from urllib import response
from xmlrpc import client
import chromadb
from dotenv import load_dotenv

from llama_index.core import ( VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document )

from llama_index.core import memory
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.schema import QueryBundle
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.readers.file import DocxReader

from networkx import nodes
from openai import OpenAI as OpenAIVision
from sqlalchemy import text

load_dotenv()

# -----------------------------
# LLM / Embeddings (global config OK)
# -----------------------------

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

class MyRagSystem:
    def __init__(self, session_id: str):
        self.session_id = session_id

        # Paths
        self.data_path = f"data/{session_id}"
        self.db_path = f"storage/{session_id}"
        self.collection_name = f"documents_{session_id}"

        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.db_path, exist_ok=True)

        # Chroma DB
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(self.collection_name)

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Memory
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        # Index
        if self.collection.count() > 0:
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        else:
            self.index = VectorStoreIndex.from_documents([], storage_context=self.storage_context)

    # -----------------------------
    # IMAGE OCR
    # -----------------------------
    def extract_text_from_image(self, image_path: str) -> str:
        client = OpenAIVision()

        with open(image_path, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode("utf-8")

        image_data_url = f"data:image/png;base64,{b64_image}"

        response = client.responses.create( model="gpt-4.1-mini",
        input=[ {  "role": "user",
                   "content": [
                    {"type": "input_text", "text": "Extract all useful information from this image."},
                    {"type": "input_image", "image_url": image_data_url}  ]  }   ]  )

        return response.output_text

    # -----------------------------
    # ASK
    # -----------------------------
    def ask(self, question, required_words="", mode="Match All", min_score=0.15):

        retriever = self.index.as_retriever(similarity_top_k=20)
        nodes = retriever.retrieve(question)

        # Similarity filter
        nodes = [n for n in nodes if getattr(n, "score", 0) >= min_score]

        # Keyword filter
        if required_words:
            keywords = [k.strip().lower() for k in required_words.split(",") if k.strip()]
            filtered = []

            for n in nodes:
                content = n.node.get_content().lower()
                if mode == "Match All":
                     if all(k in content for k in keywords):
                         filtered.append(n)
                else:
                    if any(k in content for k in keywords):
                         filtered.append(n)

            nodes = filtered
        if not nodes:
            return "No relevant documents found.", [], []

        # Rerank
        reranker = RankGPTRerank(llm=OpenAI(model="gpt-4o-mini"), top_n=3)
        reranked_nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(question))

        # Use reranked nodes
        class ManualRetriever:
            def __init__(self, nodes):
                self.nodes = nodes

            def retrieve(self, query):
                return self.nodes

        chat_engine = ContextChatEngine.from_defaults(
            retriever=ManualRetriever(reranked_nodes),
            memory=self.memory,
            system_prompt="You are a helpful assistant."  )

        response = chat_engine.chat(question)

        # Build output
        snippets = []
        sources = []
        seen = set()

        for i, n in enumerate(reranked_nodes):
            fname = n.node.metadata.get("file_name", "Unknown")

            if fname not in seen:
                sources.append((fname, round(1 - i * 0.1, 3)))
                seen.add(fname)

            snippets.append({
                "file": fname,
                "similarity_score": getattr(n, "score", 0),
                "relevance_score": 1 - i * 0.1,
                "snippet": n.node.get_content()[:250]  })

        return response.response, sources, snippets

    # -----------------------------
    # INGEST FILE
    # -----------------------------
    def ingest_file(self, path: str):
        loader = SimpleDirectoryReader(input_files=[path])
        docs = loader.load_data()

        for d in docs:
            self.index.insert(d)

    # -----------------------------
    # INGEST IMAGE
    # -----------------------------
    def ingest_image(self, path: str):
        text = self.extract_text_from_image(path)

        doc = Document( text=text,  metadata={"file_name": os.path.basename(path)} )

        self.index.insert(doc)

    # -----------------------------
    # STATS
    # -----------------------------
    def get_stats(self):
        metas = self.collection.get().get("metadatas", [])
        docs = len(set(m.get("file_name") for m in metas if m)) if metas else 0

        return { "chunks": self.collection.count(),  "documents": docs }

    # -----------------------------
    # DELETE FILE
    # -----------------------------
    def delete_single_file(self, filename: str):
        self.collection.delete(where={"file_name": filename})

        file_path = os.path.join(self.data_path, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

    # -----------------------------
    # RESET SESSION
    # -----------------------------
    def reset_full_database(self):
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass

        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.index = VectorStoreIndex.from_documents([], storage_context=self.storage_context)

        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)

        os.makedirs(self.data_path, exist_ok=True)

    # -----------------------------
    # CHECK DUPLICATE
    # -----------------------------
    def is_file_indexed(self, filename):
        results = self.collection.get(where={"file_name": filename})
        return len(results["ids"]) > 0

    # -----------------------------
    # RESET MEMORY
    # -----------------------------
    def reset_memory(self):
        self.memory.reset()

## --------------000000------------------ ##