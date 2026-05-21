# 🧠 AI Personal Retrieval Assistant 

An enterprise-grade **Agentic Retrieval-Augmented Generation (RAG)** platform designed to search, understand, reason over, and organize private documents using advanced AI orchestration.

Unlike traditional RAG chatbots, this assistant combines:

- 🧩 Multi-tool Agentic workflows
- 🔍 Multi-stage retrieval pipelines
- 🧠 Conversational memory
- 📄 Metadata-aware document discovery
- 🎯 RankGPT reranking
- 🖼️ Multimodal document ingestion
- 📚 Explainable AI evidence tracing

The result is a highly accurate and context-aware AI assistant capable of handling complex enterprise document workflows.

---

## 🚀 Live Demo

👉 Streamlit App:  
Direct Link:  
https://docsfexplo-ogjdymmw9bgqb3slvkkvgn.streamlit.app/

---

# 🌟 Platform Capabilities

## 📄 Multimodal Document Ingestion

Supports:

- PDF
- DOCX
- TXT
- XLSX
- CSV
- Images (OCR via OpenAI Vision)

Documents are automatically:
- parsed
- chunked
- embedded
- indexed into ChromaDB

---

## 🧠 Agentic RAG Architecture

The platform uses a **ReAct Agent** capable of selecting specialized tools dynamically based on user intent.

### Available Agent Tools

|---------------------------|----------------------------------|
| Tool                      | Purpose                          |
|---------------------------|----------------------------------|
| `document_search`         | Semantic question answering      |
| `document_catalog_search` | Metadata & filename discovery    |
| `spreadsheet_search`      | Structured spreadsheet reasoning |

This enables:
- semantic QA
- document title retrieval
- metadata search
- conversational workflows
- intent-aware retrieval

---

## 🎯 Adaptive Retrieval Intelligence

The assistant dynamically changes its retrieval strategy based on the user intent.
Different query types automatically trigger different retrieval depths, reranking strategies, and filtering behaviors.

### Retrieval Modes

---------------------------------------------------------------
| Query Type           | Retrieval Behavior                   |
|----------------------|--------------------------------------|
| Conversational QA    | Precision-focused semantic retrieval |
| Document Discovery   | Broad exhaustive retrieval           |
| Spreadsheet Queries  | Structured table reasoning           |
| Summaries            | Context-compressed retrieval         |
| Comparisons          | Multi-document semantic retrieval    |

### Dynamic Optimizations

- Adaptive `top_k` retrieval depth
- Dynamic RankGPT reranking depth
- Query intent classification
- Metadata-aware filtering
- Spreadsheet-specific retrieval
- Context-aware conversational retrieval

## 🔍 Multi-Stage Retrieval Pipeline

### Stage 1 — Semantic Vector Retrieval
- OpenAI embeddings
- ChromaDB similarity search

### Stage 2 — Metadata & Keyword Filtering
Supports:
- Match Any
- Match All
- hard keyword constraints

### Stage 3 — RankGPT Reranking
LLM-powered reranking improves relevance before synthesis.

### Stage 4 — LLM Synthesis
The final answer is generated using conversational context and reranked evidence.

---

## 🧠 Conversational Memory

Supports:
- multi-turn conversations
- follow-up questions
- contextual references
- session-isolated memory

Example:
> “Summarize that spreadsheet”
> “Now compare it with the previous document”

## 📊 Spreadsheet Intelligence

Unlike traditional RAG systems that flatten spreadsheets into plain text, this assistant includes a dedicated structured spreadsheet retrieval layer.

### Supported Capabilities

- Natural language querying over Excel files
- Structured row retrieval
- Price and numeric filtering
- Apartment/property search
- Table-aware reasoning
- Spreadsheet metadata filtering

### Example Queries

- "Show apartments in Manhattan"
- "List apartment prices above 1 million"
- "Find rows containing JD Edwards"
- "Show properties below 900k"
---

## 📚 Explainable AI (XAI)

Every answer includes:
- similarity scores
- relevance scores
- evidence snippets
- source tracking

This improves:
- transparency
- trustworthiness
- debugging
- enterprise auditability

---

## 📂 Session-Aware Architecture

Each session maintains:
- isolated vector collections
- independent memory
- separate uploaded documents

Supports:
- switching between sessions
- persistent retrieval
- cloud-safe state management

---

# 🏗️ Adaptive Agentic Architecture


```text
User Query
    ↓
Query Classification
    ↓
ReAct Agent
    ↓
Tool Selection
    ├── document_search
    ├── spreadsheet_search
    └── document_catalog_search
            ↓
Adaptive Retrieval Pipeline
    ↓
Dynamic Retrieval Strategy
    ├── Precision Retrieval
    ├── Exhaustive Discovery
    ├── Structured Table Search
    ↓
Metadata & Keyword Filtering
    ↓
Similarity Filtering
    ↓
RankGPT Reranking
    ↓
LLM Synthesis
    ↓
Final Answer + Evidence
```

---

# 🛠️ Technology Stack

| Layer | Technology |
|------|-------------|
| Frontend | Streamlit |
| Orchestration | LlamaIndex |
| Agent Framework | ReActAgent |
| LLM Provider | OpenAI |
| Vector Database | ChromaDB |
| OCR / Vision | OpenAI Vision |
| Embeddings | text-embedding-3-small |
| Language | Python |

---

# 🤖 Models Used

| Purpose               | Model                  |
|-----------------------|------------------------|
| Agent Reasoning       | GPT-4.1-mini           |
| Retrieval & Synthesis | GPT-4.1-mini           |
| OCR / Fast Tasks      | GPT-4.1-nano           |
| Embeddings            | text-embedding-3-small |
| Reranking             | RankGPT                |

---

# 🔍 Example Queries

### 📄 Document Discovery

- "List all JD Edwards job descriptions"
- "Which files mention EnterpriseOne?"
- "Show documents related to AS400"

### 📊 Spreadsheet Analysis

- "Show apartments in Manhattan"
- "List apartment prices above 1 million"
- "Find rows containing Foxter"

### 🧠 Conversational QA

- "Summarize the Johnson&Johnson job description"
- "Compare the CNC responsibilities between documents"
- "What JDE modules are required?"

---
 
# 🚀 Quick Start

## 1️⃣ Clone Repository

```bash
git clone https://github.com/luisrvq/docs_fexplo.git
cd docs_fexplo
```

---

## 2️⃣ Create Virtual Environment

### Mac / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

# ▶️ Run Locally

```bash
streamlit run maing.py
```

---

# ☁️ Streamlit Cloud Deployment

## 1️⃣ Push Code to GitHub

## 2️⃣ Open Streamlit Community Cloud

https://streamlit.io/cloud

## 3️⃣ Create New App

Set:

```text
Main file:
maing.py
```

---

## 4️⃣ Add Secrets

```toml
OPENAI_API_KEY = "your_api_key_here"
```

---

# 📁 Project Structure

```text
.
├── data/                      # Uploaded runtime documents
├── storage/                   # Persistent vector storage
├── maing.py                   # Streamlit UI + session management
├── agent_manager.py           # Agent orchestration & tools
├── rag_pipelineg.py           # Core RAG pipeline
├── requirements.txt
├── faqs_maing.json            # FAQ retrieval source
├── README.md
└── assets/
    ├── maing_sidebar.png
    └── maing_chat.png
```

---

# 🔍 Retrieval Features

✅ Semantic Search  
✅ Metadata Filtering  
✅ Hard Keyword Constraints  
✅ RankGPT Reranking  
✅ Multi-Document Retrieval  
✅ Session-Isolated Retrieval  
✅ Explainable Evidence Tracking  

---

# 📸 Screenshots

## Sidebar & Controls

![Sidebar](assets/maing_sidebar.png)

---

## Chat & Evidence

![Chat](assets/maing_chat.png)

---

# ⚠️ Notes

- Local desktop features may not work in cloud environments
- Very large files may increase ingestion time
- OCR quality depends on source image quality
- Streamlit Cloud storage is ephemeral

---

# 🔮 Future Roadmap

- [ ] Hybrid Search (BM25 + Vector Retrieval)
- [ ] Workflow-based ReActAgent Migration

---

# 👤 Author

Luis Solis Vasquez

---

# ⭐ Support the Project

If you found this project useful, please consider giving it a ⭐ on GitHub.
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push your code to GitHub
2. Go to: https://streamlit.io/cloud
3. Click **"New app"**
4. Select your repo and branch
5. Set the main file:

   ```
   maing.py
   ```
6. Add secrets:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

7. Deploy 🚀

---

## 🔮 Future Improvements

- [ ] **Hybrid Search**: Combining BM25 (keyword) with Vector search.
- [ ] **Local LLM Support**: Integration with Ollama for privacy-focused queries.
- [ ] **Streaming Responses**: For a more "ChatGPT-like" typing experience.

---

## 👤 Author
Luis Solis Vasquez

---

## ⭐ If you like this project or find it helpful, please give it a star! **

