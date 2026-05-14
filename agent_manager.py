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
        
        # Pull settings directly from Streamlit session state
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

            return answer   # Return ONLY text to the Agent
        
        # 2. Define the RAG as a Tool
        # This tells the Agent: "If you need facts from docs, call this."
        self.rag_tool = FunctionTool.from_defaults(
            fn=search_docs,
            name="document_search",
            description="""
                            Search uploaded documents filename and contents.

                            Use this tool when:
                            - the user mentions uploaded files
                            - the user references filenames
                            - the user asks for summaries
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
        
        # 3. Initialize the Agent with the tool(s)
        # We use gpt-4o because it's better at "ReAct" reasoning
        ##self.agent = ReActAgent.from_tools(
        self.agent = ReActAgent(
            tools=[self.rag_tool], 
            llm=OpenAI(model="gpt-4o"),
            memory=self.rag_system.memory,
            verbose=True  )

    def handle_query(self, user_input):
        # This is the single entry point for your Streamlit UI
        response = self.agent.chat(user_input)
        
        # Access the RAG's metadata for your UI sources
        # (This pulls from the tool's execution history)
        sources = getattr(self.rag_system, 'last_sources', []) 
        snippets = getattr(self.rag_system, 'last_snippets', [])
        
        #return response.response, sources, snippets
        return str(response), sources, snippets
## --------------------------- 00 --------------------------- ##