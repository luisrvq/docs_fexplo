# rag_pipelineg.py
#-----------------
import re
import json
import numpy
from csv import excel
from importlib.resources import path
from operator import index
import os
import shutil
import base64
import pandas as pd        # For Excel processing
from unittest import loader
from urllib import response
from xmlrpc import client
import chromadb
from dotenv import load_dotenv
from llama_index.core import ( VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document )
from llama_index.core import memory
from llama_index.core import chat_engine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.readers.file import DocxReader
from llama_index.readers.file import PyMuPDFReader

from networkx import nodes
from openai import OpenAI as OpenAIVision
from sqlalchemy import text

load_dotenv()

# ---------------------------------------------------------------------------------------------
# AI Models (global config) getting from TOML or .env 
# is optional since you can also set them directly in the code, 
# but it's a good practice to keep them in environment variables for security and flexibility.
# ---------------------------------------------------------------------------------------------
MODEL_REASONING  = os.getenv( "OPENAI_MODEL_REASONING", "gpt-4.1-mini" )  # More powerful model for reasoning tasks
MODEL_EMBEDDINGS = os.getenv( "OPENAI_MODEL_EMBEDDINGS", "text-embedding-3-small" )  # More cost-effective embedding model
MODEL_RETRIEVAL  = os.getenv( "OPENAI_MODEL_RETRIEVAL", "gpt-3.5-turbo" )  # More cost-effective model for retrieval tasks
MODEL_IMAGING    = os.getenv( "OPENAI_MODEL_IMAGING", "gpt-4.1-mini" )  # Model for image analysis (if needed)

# -----------------------------------
# LLM / Embeddings (global config OK)
# -----------------------------------

Settings.llm = OpenAI(model=MODEL_REASONING)
Settings.embed_model = OpenAIEmbedding(model=MODEL_EMBEDDINGS)
Settings.text_splitter = SentenceSplitter( chunk_size=700, chunk_overlap=80 )

class MyRagSystem:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.spreadsheets = {}               # For Excel processing        
        self.last_spreadsheet_results = []   # For Excel processing

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
            self.reload_spreadsheet_store()  # Load spreadsheets into memory on initialization
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

        response = client.responses.create( model=MODEL_IMAGING,
        input=[ {  "role": "user",
                   "content": [
                    {"type": "input_text", "text": "Extract all useful information from this image."},
                    {"type": "input_image", "image_url": image_data_url}  ]  }   ]  )

        return response.output_text

    # -----------------------------
    # ASK
    # -----------------------------
    def ask(self, question, required_words="", mode="Match All", min_score=0.15, query_type=None):
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.chat_engine import CondensePlusContextChatEngine
        from llama_index.core.postprocessor import SimilarityPostprocessor
        from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
        from llama_index.llms.openai import OpenAI   

        metadata_filters = None
        if "faq" not in question.lower():
            metadata_filters = MetadataFilters( filters=[ ExactMatchFilter( key="type", value="user_upload" ) ] )
            
        # Spreadsheet-focused retrieval
        if query_type == "spreadsheet_query":
            metadata_filters = MetadataFilters( filters=[ ExactMatchFilter( key="data_type", value="structured_table" ) ] )
            print("Spreadsheet filter activated")

        # 2. Define the Base Retriever (Dynamic Retrieval Strategy)
        if query_type == "retrieval_list":         
            top_k = 12
            top_n = 5
            effective_min_score = min(min_score, 0.10)
        elif query_type == "comparison":           
            top_k = 12
            top_n = 8
            effective_min_score = min_score
        elif query_type == "summary":              
            top_k = 10
            top_n = 5
            effective_min_score = min_score
        elif query_type == "image_query":          
            top_k = 6
            top_n = 3
            effective_min_score = min_score
        else:                                      
            top_k = 12
            top_n = 5
            effective_min_score = min_score
        print("\n===== RETRIEVAL CONFIG =====")
        print(f"Query Type: {query_type}")
        print(f"top_k: {top_k}")
        print(f"top_n: {top_n}")
        print(f"min_score: {effective_min_score}")
        
        base_retriever = self.index.as_retriever(similarity_top_k=top_k, filters=metadata_filters)
        print(type(metadata_filters))
        print(metadata_filters)
        print(f"Using top_k = {top_k}")

        # 3. Prepare Post-Processors (Filters + Reranker)
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=effective_min_score)]
        # Keyword Filter logic remains manual because it depends on user input strings
        # We will apply it to the final response nodes below.
    
        # LLM Reranker (Top 5)
        reranker = RankGPTRerank(llm=OpenAI(model=MODEL_RETRIEVAL), top_n=top_n)
        postprocessors.append(reranker)

        # 4. Build the Query Engine (Fixed Initialization)
        # Using from_args allows easier configuration of the response synthesizer
        query_engine = RetrieverQueryEngine.from_args( retriever=base_retriever, node_postprocessors=postprocessors,
                        response_mode="compact"  )

        # 5. Get response
        response = query_engine.query(question)

        # 7. Extract nodes for your UI sources (using nodes from the response)
        # This ensures your "Sources & Evidence" match what the AI actually used
        final_nodes = response.source_nodes[:10]  # Extract only the 3 top documents
        # 8. Temporary for debugging: print out the retrieved nodes and their metadata to verify what's being processed
        print("\n===== RETRIEVED NODES =====")
        for i, n in enumerate(response.source_nodes[:top_n]): # Print top_n for debugging
            print(f"\nNODE {i}")
            print(f"Score: {n.score}")
            print(f"Metadata: {n.node.metadata}")
            print(n.node.get_content()[:50])

        snippets = []
        sources = []
        seen = set()
    
        for i, n in enumerate(final_nodes):
            fname = n.node.metadata.get("file_name", "Unknown Document")
            # Manual Keyword Filter logic (LlamaIndex doesn't have a native one-to-one for your mode)
            if required_words:
                keywords = [k.strip().lower() for k in required_words.split(",") if k.strip()]
                content = n.node.get_content().lower()
                if mode == "Match All" and not all(k in content for k in keywords):
                    continue
                if mode == "Match Any" and not any(k in content for k in keywords):
                    continue

            if fname not in seen:
                score = round(n.score, 3) if n.score is not None else 1.0
                sources.append((fname, score))
                seen.add(fname)
                snippets.append({ "file": fname, "similarity_score": getattr(n, "score", 0),  "relevance_score": 1 - i * 0.1, 
                                  "snippet": n.node.get_content()[:300] + "..." })

        self.last_sources = sources   # To make sure those sources actually show up in your UI
        self.last_snippets = snippets
        
        return str(response), sources, snippets

    # -----------------------------
    # INGEST FILE
    # -----------------------------
    def ingest_file(self, path: str):
        file_name = os.path.basename(path)   # Get just the filename for metadata and storage 
        ## PDF  
        if path.lower().endswith(".pdf"):
            loader = PyMuPDFReader()
            docs = loader.load_data(file_path=path)
        ## DOCX
        elif path.lower().endswith(".docx"):
            loader = DocxReader()
            docs = loader.load_data(file=path)  # Load the DOCX file into documents
        ## TXT, PNG, JPG, etc.
        else:
            loader = SimpleDirectoryReader(input_files=[path])
            docs = loader.load_data()   
        
        print(f"Loaded {len(docs)} docs from {path}")  # Only for debugging, can be removed in production
        for i, d in enumerate(docs):                   # Only for debugging, can be removed in production
            print(f"\n--- DOC {i} CONTENT PREVIEW ---")
            print(d.text[:1000])

        for d in docs:
            d.metadata["file_name"] = os.path.basename(path)
            d.metadata["type"] = "user_upload"
            self.index.insert(d)

    # -----------------------------
    # INGEST IMAGE
    # -----------------------------
    def ingest_image(self, path: str):
        text = self.extract_text_from_image(path)

        doc = Document( text=text,  metadata={"file_name": os.path.basename(path), "type": "user_upload"} )
        self.index.insert(doc)

    # -----------------------------
    # INGEST Excel files
    # -----------------------------
    def load_excel_all_sheets(self, path, filename):
        from llama_index.core import Document
        import pandas as pd
        excel = pd.ExcelFile(path)
        documents = []
        for sheet_name in excel.sheet_names: 
            df = excel.parse(sheet_name) 
            
            markdown_table = df.to_markdown(index=False)    # Convert the entire sheet to a Markdown table
            content = f"### Sheet: {sheet_name}\n\n{markdown_table}"  # wrap the table in a clear header so the AI knows which sheet it is reading
            doc = Document( text=content, metadata={ "file_name": filename, "sheet_name": sheet_name, 
                                                    "data_type": "structured_table", "source_type": "excel", "type": "user_upload" } ) 
            documents.append(doc)

        return documents


    # -----------------------------
    # STATS
    # -----------------------------
    def get_stats(self):
        metas = self.collection.get().get("metadatas", [])
        user_docs = set()  # User uploaded documents only
        faq_docs = set()   # FAQ documents
        for m in metas:
            if not m: 
                continue
            file_name = m.get("file_name", "Unknown")
            doc_type = m.get("type", "user_upload") 
            if doc_type == "official_faq":
                faq_docs.add(file_name)
            else:
                user_docs.add(file_name)

        return {
            "chunks": self.collection.count(),
            "documents": len(user_docs),
            "faq_documents": len(faq_docs),
            "total_documents": len(user_docs) + len(faq_docs)  }

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

    # -----------------------------
    # Load faqs_maing.json (FAQs)
    # -----------------------------
    def load_faqs(self, json_file_path):
        import json
        with open(json_file_path, 'r') as f:
            faq_data = json.load(f)
            faq_documents = []
            for item in faq_data:
                # We put the question in the main text so the Vector Search finds it
                # We put the answer in the metadata or text so the LLM can read it
                doc_content = f"Question: {item['question']}\nAnswer: {item['answer']}" 
                doc = Document( text=doc_content, 
                           metadata={
                                    "file_name": "FAQs",
                                    "category": item["category"],
                                    "type": "official_faq"  }  )
                faq_documents.append(doc)
        return faq_documents
    
    # --------------------------------------------------------
    # Search Document Titles (with optional keyword filtering)
    # --------------------------------------------------------
    def search_document_titles(self, query, required_words="", mode="Match Any", min_score=0.20, top_k=30 ):
    # Build metadata filters
        metadata_filters = MetadataFilters( filters=[ ExactMatchFilter( key="type", value="user_upload" ) ] )
        # Retriever
        retriever = self.index.as_retriever( similarity_top_k=top_k, filters=metadata_filters )
        # Retrieve nodes
        nodes = retriever.retrieve(query)
        # Filter by similarity
        filtered_nodes = [ n for n in nodes if n.score and n.score >= min_score ]
        # Extract unique filenames
        unique_files = sorted(set( n.node.metadata.get("file_name", "Unknown") for n in filtered_nodes ))
        # Optional keyword filtering
        if required_words:
            keywords = [ k.strip().lower() for k in required_words.split(",") ]
            filtered_files = []
            for f in unique_files:
                f_lower = f.lower()
                if mode == "Match All":
                    if all(k in f_lower for k in keywords):
                        filtered_files.append(f)
                else:
                    if any(k in f_lower for k in keywords):
                        filtered_files.append(f)
            unique_files = filtered_files

        # Return formatted result
        if not unique_files:
            return "No matching documents found."
        return "\n".join(
            f"- {f}" for f in unique_files )
    # ------------------------------------------------------------------------------------
    # Load Excel/CSV into self.spreadsheets WITHOUT indexing (for spreadsheet_search tool)
    # ------------------------------------------------------------------------------------
    def load_spreadsheet_store(self, path: str, filename: str):
        """Load Excel/CSV into self.spreadsheets WITHOUT indexing."""
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            self.spreadsheets[filename] = df
            print(f"Spreadsheet stored (no indexing): {filename}")
            print("Spreadsheet keys now:", list(self.spreadsheets.keys()))
        except Exception as e:
            print(f"Spreadsheet store failed: {e}")
            raise
    # ------------------------------------------------------------------------------------
    # ReLoad Excel/CSV into self.spreadsheets WITHOUT indexing (for spreadsheet_search tool)
    # ------------------------------------------------------------------------------------
    def reload_spreadsheet_store(self):
        self.spreadsheets = {}
        self.spreadsheet_schema = {}
        upload_dir = self.data_path
        for filename in os.listdir(upload_dir):
            if filename.lower().endswith((".xlsx", ".xls", ".csv")):
                path = os.path.join(upload_dir, filename)
                self.load_spreadsheet_store(path, filename)

    # -----------------------------------------------------------------------------------------------------------
    # New Parser with LLM. It will convert the user query into a structured JSON format that specifies 
    # which columns to filter on, what operators to use, and what values to compare against. 
    # This structured format can then be used by your spreadsheet_search tool to perform the actual data retrieval 
    # from the loaded spreadsheets.   
    # ------------------------------------------------------------------------------------------------------------
    def parse_query_with_llm(self, query: str, available_columns: list):

        prompt = f"""
    You are an expert spreadsheet query parser.

    Available columns:
    {available_columns}

    Convert the user request into JSON.

    Return ONLY valid JSON.

    Do NOT use markdown.
    Do NOT use ```json.
    Do NOT provide explanations.

    Schema:
    {{
    "query_type": 
        "spreadsheet|document_search|document_list|general",
    "filters": [
        {{
        "column": "column name",
        "operator": ">|<|>=|<=|=|contains",
        "value": "value"
        }} ],

    "columns": ["column1", "column2" ],

    "sort": {{"column": "column name", "direction": "asc|desc" }},
      
    "search_terms": ["term1", "term2"] 
    }}

    Determine the query type.

    Use:

    spreadsheet:
    - apartments
    - apartment listings
    - property listings
    - excel
    - csv
    - rows
    - columns
    - table data
    - filtering spreadsheet data

    document_list:
    - list document titles
    - list filenames
    - uploaded files
    - show documents
    - what documents do I have

    document_search:
    - find documents mentioning
    - search resumes
    - search job descriptions
    - search uploaded documents

    general:
    - conversational questions
    - explanations
    - advice

    For numeric values return numbers without quotes.
    GOOD: "value": 81
    BAD: "value": "81"
    Return ONLY valid JSON.

    For ranges:
    Examples:
    "area between 80 and 120 m2"
    Return:
    "column": "Area (M2)",
    "operator": "between",
    "value": [80,120]

    "price between 900000 and 1200000"
    Return:
    "column": "List Price",
    "operator": "between",
    "value": [900000,1200000]

    For sorting:
    Examples:
    "cheapest apartments"
    Return:
    "sort": 
        "column":"List Price",
        "direction":"asc"

    "most expensive apartments"
    Return:
    "sort": 
        "column":"List Price",
        "direction":"desc"

    "newest apartments"
    Return:
    "sort": 
        "column":"Year",
        "direction":"desc"

    "oldest apartments"
    Return:
    "sort":
        "column":"Year",
        "direction":"asc"
   

    User request:
    {query}
    """

        llm = OpenAI(model=MODEL_REASONING)

        response = llm.complete(prompt)

        try:
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()    
            return json.loads(response_text)
       
        except Exception as e:
            print("JSON Parse Error:", e)
            print(response)
            return {
                "query_type": "general",
                "filters": [],
                "columns": [],
                "sort": { },
                "search_terms": []
            }  
        
    # ----------------------------------------------------------------------------------------------
    # This tool function will decide what to filter based on the JSON Parser. 
    # Apply LLM-generated filters to a dataframe 
    # ----------------------------------------------------------------------------------------------  
    def apply_dataframe_filters(self, df, filters):
        """
        Apply LLM-generated filters to a dataframe.
        Supported operators:
        =, contains, >, <, >=, <=, between
        """

        working_df = df.copy()

        for f in filters:

            try:
                column = f.get("column")
                operator = f.get("operator")
                value = f.get("value")

                if column not in working_df.columns:
                    print(f"Column not found: {column}")
                    continue

                if operator in [">", "<", ">=", "<=", "between"]:     # Numeric operators
                    numeric_series = pd.to_numeric(working_df[column],  errors="coerce" )
                    if operator == ">":
                        working_df = working_df[numeric_series > float(value)]
                    elif operator == "<":
                        working_df = working_df[numeric_series < float(value)]
                    elif operator == ">=":
                        working_df = working_df[numeric_series >= float(value)]
                    elif operator == "<=":
                        working_df = working_df[numeric_series <= float(value)]
                    elif operator == "between":
                        low, high = value
                        working_df = working_df[ numeric_series.between( float(low), float(high) ) ]

                elif operator == "=":                     # Exact Match
                    working_df = working_df[ working_df[column].astype(str).str.lower() == str(value).lower() ]
                
                elif operator == "contains":              # Contains
                    working_df = working_df[ working_df[column].astype(str).str.contains(str(value), case=False, na=False ) ]

                elif operator == "between":               # Between (for numeric ranges)
                        if ( isinstance(value, list) and len(value) == 2 ):
                            low, high = value
                            working_df = working_df[numeric_series.between( float(low), float(high) ) ]

            except Exception as e:
                print(f"Filter error: {e}")
                print( f"Filter failed "  f"{column} {operator} {value}" )
                continue

        return working_df
    
    # ----------------------------------------------------------------------------------------------
    # This tool function will sort the dataframe 
    # Apply LLM-generated filters to a dataframe 
    # ----------------------------------------------------------------------------------------------  
    def apply_sorting( self, df, sort_config):

        if not sort_config:
            return df

        column = sort_config.get("column")
        direction = sort_config.get("direction", "asc")
        if column not in df.columns:
            print( f"Sort column not found: {column}" )
            return df
        ascending = ( direction.lower() == "asc" )

        return df.sort_values( by=column, ascending=ascending )

    # ----------------------------------------------------------------------------------------------
    # Column Selection Helper 
    # ----------------------------------------------------------------------------------------------  
    def select_columns(self, df, columns):
        if not columns:
            return df
        valid_cols = [c for c in columns if c in df.columns ]
        return df[valid_cols]
## --------------------------- 000000000 --------------------------- ##