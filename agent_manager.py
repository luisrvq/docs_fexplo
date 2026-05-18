# agent_manager.py
#-----------------

import pandas as pd
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import streamlit as st
from rag_pipelineg import MyRagSystem   # Import my existing class

class AssistantAgent:
    def __init__(self, rag_instance):

        from rag_pipelineg import MyRagSystem   # Import my existing class
        # 1. Initialize your existing stable RAG
        self.rag_system = rag_instance  # Use the existing RAG instance passed from Streamlit session state
        
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
                min_score=st.session_state.get("min_score_key", 0.15),  )

            self.rag_system.last_sources = sources   # Store sources for UI
            self.rag_system.last_snippets = snippets
            print("TOOL USED: document_search")
            return answer   # Return ONLY text to the Agent
        

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
        
        # This is another separate tool that ONLY searches inside spreadsheets, 
        # which can be useful for queries that specifically reference tables, Excel files, or CSVs.
        def spreadsheet_tool_fn(query, max_rows=20):
            query_lower = query.lower()
            results = []
            numeric_filter = self.rag_system.parse_numeric_query(query)
            for file_name, df in self.rag_system.spreadsheets.items():
                try:
                    working_df = df.copy()
                    # NUMERIC FILTERING
                    # ----------------------------
                    if numeric_filter:
                        # Detect likely price columns
                        price_columns = [
                            c for c in working_df.columns
                            if any(keyword in str(c).lower() for keyword in [ "price", "value", "amount", "cost" ]) ]
                        for col in price_columns:
                            try:
                                numeric_series = pd.to_numeric( working_df[col], errors="coerce" )
                                if numeric_filter["operator"] == "gt":
                                    filtered_df = working_df[ numeric_series > numeric_filter["value"] ]
                                elif numeric_filter["operator"] == "lt":
                                    filtered_df = working_df[ numeric_series < numeric_filter["value"] ]
                                else:
                                    continue

                                if not filtered_df.empty:
                                    results.append({ "file": file_name, "rows": filtered_df.head(max_rows)  })

                            except Exception as e:
                                print(f"Numeric filter failed for column {col}: {e}")

                    # TEXT FILTERING
                    # ----------------------------
                    else:
                        mask = working_df.astype(str).apply(
                            lambda col: col.str.lower().str.contains( query_lower, na=False ) )
                        matched_rows = working_df[mask.any(axis=1)]
                        if not matched_rows.empty:
                            results.append({ "file": file_name, "rows": matched_rows.head(max_rows) })
                except Exception as e:
                    print(f"Spreadsheet search error in {file_name}: {e}")

            ##self.last_spreadsheet_results = results
            if not results:
                return "No spreadsheet matches found."
            final_output = []
            for r in results:
                final_output.append( f"\\n📄 FILE: {r['file']}\\n" )
                final_output.append( r["rows"].to_string(index=False) )

            return "\\n\\n".join(final_output)
        
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
            llm=OpenAI(model="gpt-3.5-turbo"),
            memory=self.rag_system.memory,
            verbose=True  )

    def handle_query(self, user_input):
        try:
            # This is the single entry point for your Streamlit UI
            response = self.agent.chat(user_input)
            output_text = response.response
        except Exception as e:
            # 🛡️ Fallback: If gpt-3.5 reasoning layer fails, bypass agent 
            # and query your stable RAG directly so the user gets an answer!
            print(f"Agent Reasoning Loop Failed: {str(e)}. Routing to direct RAG.")
            output_text, _, _ = self.rag_system.ask(
                user_input,
                required_words=st.session_state.get("keyword_filter_key", ""),
                mode=st.session_state.get("logic_mode_key", "Match All"),
                min_score=st.session_state.get("min_score", 0.15),
                reset_memory=False  )

        # Access the RAG's metadata for your UI sources
        # (This pulls from the tool's execution history)
        sources = getattr(self.rag_system, 'last_sources', []) 
        snippets = getattr(self.rag_system, 'last_snippets', [])
        
        #return response.response, sources, snippets
        return output_text, sources, snippets
## --------------------------- 00 --------------------------- ##