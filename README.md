# 🧠 AI Personal Assistant (Agentic Multimodal RAG)

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
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://maingpy-6tg79pvqkn8plxpn7lsxsc.streamlit.app)

Direct Link:  
https://maingpy-6tg79pvqkn8plxpn7lsxsc.streamlit.app

---

# 🌟 Core Features

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

| Tool | Purpose |
|------|---------|
| `document_search` | Semantic question answering |
| `document_catalog_search` | Metadata & filename discovery |

This enables:
- semantic QA
- document title retrieval
- metadata search
- conversational workflows
- intent-aware retrieval

---

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

# 🏗️ Architecture Overview

```text
User Query
    ↓
ReAct Agent
    ↓
Tool Selection
    ├── document_search
    └── document_catalog_search
            ↓
Multi-Stage Retrieval Pipeline
    ↓
Vector Search (ChromaDB)
    ↓
Keyword + Metadata Filtering
    ↓
RankGPT Reranking
    ↓
LLM Synthesis
    ↓
Final Answer + Sources + Evidence
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

| Purpose | Model |
|---------|------|
| Agent Reasoning | GPT-3.5-Turbo |
| Retrieval Synthesis | GPT-4-Turbo |
| OCR / Vision | GPT-4o-mini |
| Embeddings | text-embedding-3-small |
| Reranking | RankGPT |

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

## 5️⃣ Deploy 🚀

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

- [ ] Local LLM Support (Ollama)
- [ ] Spreadsheet-Specific Agent

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

## 📁 Project Structure

```
.
├── app/                 # UI components, helpers
├── data/                # Uploaded documents (runtime)
├── .gitkeep
├── maing.py             # Streamlit UI, session state management, and file handling
├── rag_pipelineg.py     # The core RAG logic, custom retrievers, and reranking post-processors
├── requirements.txt
├── .gitignore
├── faqs_maing.json      # Pre-vetted Q&A pairs for high-precision retrieval
└── README.md
```

---

## ⚠️ Notes

* “Open Folder” works only locally (desktop environment)
* In cloud, files are available via **Download button**
* Large files may impact performance

---

## 🔮 Future Improvements

- [ ] **Hybrid Search**: Combining BM25 (keyword) with Vector search.
- [ ] **Local LLM Support**: Integration with Ollama for privacy-focused queries.
- [ ] **Streaming Responses**: For a more "ChatGPT-like" typing experience.

---
## 📸 Screenshots

### Sidebar & Controls
![Sidebar](assets/maing_sidebar.png)

### Chat & Sources
![Chat](assets/maing_chat.png)

---

## 👤 Author
Luis Solis Vasquez

---

## ⭐ If you like this project or find it helpful, please give it a star! **

