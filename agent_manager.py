# agent_manager.py
#-----------------

import pandas as pd
import re
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import streamlit as st
from rag_pipelineg import MyRagSystem   # Import my existing class
from rag_pipelineg import MODEL_RETRIEVAL

class AssistantAgent:
    def __init__(self, rag_instance):

        from rag_pipelineg import MyRagSystem   # Import my existing class
        # 1. Initialize your existing stable RAG
        self.rag_system = rag_instance  # Use the existing RAG instance passed from Streamlit session state

        #---------------------------------------------------------------------------------------------------
        # Define the function that will be wrapped as a tool for the Agent. 
        # This function will call your RAG's ask() method.
        #------------------------------------------------------------------ 
        def search_docs(query: str):
            """Search for information in uploaded documents and spreadsheets."""
            # Pull settings directly from Streamlit session state
            answer, sources, snippets = self.rag_system.ask(
                query, 
                required_words=st.session_state.get("keyword_filter_key", ""),
                mode=st.session_state.get("logic_mode_key", "Match All"),
                min_score=st.session_state.get("min_score_key", 0.15), query_type=None  )

            self.rag_system.last_sources = sources   # Store sources for UI
            self.rag_system.last_snippets = snippets
            print("TOOL USED: document_search. This is deactivating the Agent's ability to call the RAG directly, but you can re-enable if you want the Agent to have more autonomy in deciding when to search documents.")
            return answer   # Return ONLY text to the Agent
        
        #---------------------------------------------------------------------------------------------------
        # This is a separate tool that ONLY searches document titles and filenames, 
        # which can be useful for certain queries that reference file names or ask "What documents do I have
        #---------------------------------------------------------------------------------------------------
        def search_document_titles(query: str):
            """Search ONLY for matching document titles and filenames."""
            result = self.rag_system.search_document_titles( query, 
                                                        required_words=st.session_state.get( "keyword_filter_key", "" ), 
                                                        mode=st.session_state.get( "logic_mode_key", "Match Any" ), 
                                                        min_score=st.session_state.get( "min_score_key", 0.15 ) )
            print("TOOL USED: document_catalog_search")
            return result
        #----------------------------------------------------------------------------------------
        # Use LLM to parser the prompt and extract filters and column selections, then apply to each spreadsheet.
        #----------------------------------------------------------------------------------
        def spreadsheet_tool_fn(query, max_rows=30, parsed_query=None):
            query_lower = query.lower()
            results = []

            # 3. Loop through spreadsheets
            #-----------------------------
            for file_name, df in self.rag_system.spreadsheets.items():
                print(f"Processing spreadsheet: {file_name}")  # Debug print to check which file is being processed
                working_df = df.copy()  

                # parse prompt using LLM to extract filters and column selections, then apply to each spreadsheet
                #------------------------------------------------------------------------------------------------
                #parsed_query = self.rag_system.parse_query_with_llm( query, list(working_df.columns) )  
                print( f"LLM-parsed query structure: {parsed_query}" )  # Debug print to check LLM parsing output

                working_df = (self.rag_system.apply_dataframe_filters( working_df, parsed_query["filters"] ) )

                working_df = ( self.rag_system.apply_sorting( working_df, parsed_query.get("sort", {} ) ) )
                print( "\n===== SORT =====" )
                print( parsed_query.get( "sort", {} ) )
                print( "================\n")
                
                working_df = (self.rag_system.select_columns( working_df, parsed_query["columns"] ) )

                # Append results
                # -----------------------------------------
                if not working_df.empty:
                    results.append({"file": file_name,
                                    "rows": working_df.head(max_rows) }) 
                    return results
                else:   
                    return "No spreadsheet matches found."

        #----------------------------------------------------------------
        # 2a. Define the RAG as a Tool
        # This tells the Agent: "If you need facts from docs, call this."
        #----------------------------------------------------------------
        self.rag_tool = FunctionTool.from_defaults(
            fn=search_docs,
            name="document_search",
            description="""
                            Search uploaded documents filename and contents.

                            Use this tool when:
                            - the user mentions uploaded files
                            - the user asks about document contents
                            - the user wants summaries
                            - the user asks for bullet lists
                            - the user asks about spreadsheets
                            - the user asks about PDFs or images

                            The tool can search:
                            - PDFs
                            - Excel spreadsheets
                            - Word documents
                            - OCR extracted images
                            - FAQs
                            """  )
        # 2b. Define a separate tool for searching document titles only. 
        # This can help the Agent decide when to pull up file names vs. when to pull content.
        #------------------------------------------------------------------------------------
        self.catalog_tool = FunctionTool.from_defaults(
            fn=search_document_titles,
            name="document_catalog_search",
            description="""
            Use this tool when the user asks for:
            - document titles
            - filenames
            - uploaded files
            - resumes
            - PDFs matching a topic
            - job descriptions by category
            - document lists

            IMPORTANT:
            This tool returns ONLY document names,
            not document contents.

            Use this tool for:
            - "list documents"
            - "find resumes"
            - "show filenames"
            - "documents about EnterpriseOne"
            - "job description files"
            """
        )

        self.spreadsheet_tool = FunctionTool.from_defaults(
            fn=spreadsheet_tool_fn,
            name="spreadsheet_search",
            description="""
            Use this tool for:

            - Excel files
            - CSV files
            - apartments
            - prices
            - tabular data
            - rows and columns
            - spreadsheet filtering
            - structured data extraction
            - property listings
            - inventory lists
            - customer tables

            Use this tool when the user asks:

            - list apartments
            - show rows
            - filter spreadsheet
            - search Excel
            - extract tabular data
            - find records
            - summarize spreadsheet data
            - locate spreadsheet values
            - make calculations with spreadsheet data

            IMPORTANT:
            This tool works specifically with structured spreadsheet data.
            """
        )

        # 3. Initialize the Agent with the tool(s)
        # We use gpt-4-turbo-preview because it's better at "ReAct" reasoning
        self.agent = ReActAgent(
            tools=[self.catalog_tool, 
                   self.rag_tool, 
                   self.spreadsheet_tool], 
            llm=OpenAI(model=MODEL_RETRIEVAL),
            memory=self.rag_system.memory,
            verbose=True  )

    #-----------------------------------------------------------------------------------------------------------
    # handle_query optimized to use LLM parsing to detect numeric/spreadsheet intent and relevant Excel data 
    # before routing to the Agent,
    #-----------------------------------------------------------------------------------------------------------
    def handle_query(self, user_input):
        # No spreadsheets loaded?

        available_columns = []
        if self.rag_system.spreadsheets:
            first_df = next( iter(self.rag_system.spreadsheets.values()) )
            available_columns = list( first_df.columns )

        parsed = self.rag_system.parse_query_with_llm( user_input, available_columns )

        print("\n===== PARSED QUERY =====")
        print(parsed)
        print("========================\n")

        query_type = parsed.get( "query_type", "general" )

        # Spreadsheet route
        #------------------
        if query_type == "spreadsheet":
            print( "🔧 Routing to spreadsheet_search" )
            try:
                results = self.spreadsheet_tool.fn( user_input, parsed_query=parsed )
                return results, [], []
            except Exception as e:
                print(  f"Spreadsheet tool failed: {e}"  )
                return ( "Spreadsheet search failed.", [], []  )        
        elif query_type == "document_list":
            print( "📄 Routing to catalog_tool" )
            result = self.catalog_tool.fn( user_input )
            return result, [], []
        elif query_type == "document_search":
            print( "📚 Routing to RAG" )
            return self.rag_system.ask(
                user_input,
                required_words=st.session_state.get(
                    "keyword_filter_key",
                    "" ),
                mode=st.session_state.get(
                    "logic_mode_key",
                    "Match All" ),
                min_score=st.session_state.get(
                    "min_score_key",
                    0.15 ),
                query_type=query_type  )
        else:
            print( "💬 Routing to general RAG"  )
            return self.rag_system.ask(
                user_input,
                required_words=st.session_state.get(
                    "keyword_filter_key",
                    ""  ),
                mode=st.session_state.get(
                    "logic_mode_key",
                    "Match All"  ),
                min_score=st.session_state.get(
                    "min_score_key",
                    0.15  ), 
                query_type=query_type  )

## --------------------------- 00 --------------------------- ##