# 🧠 AI RAG Personal Assistant (Multimodal)

An interactive **Retrieval-Augmented Generation (RAG)** application built with Streamlit, LlamaIndex, and OpenAI.

This assistant allows users to upload documents and images, query their content using natural language, and receive answers with **sources, evidence, and explainability scores**.

---

## 🚀 Features

* 📄 Upload and index documents (PDF, DOCX, TXT)
* 🖼️ Image understanding (PNG, JPG) using OpenAI Vision
* 🔍 Semantic search with vector embeddings
* 🧠 Context-aware chat (follow-up questions supported)
* 📊 Dual scoring:

  * **Similarity score** (retriever)
  * **Relevance score** (reranker)
* 🎯 Keyword filtering (Match All / Match Any)
* ⚙️ Adjustable similarity threshold
* 📂 Document management UI (preview, delete, download)
* ☁️ Deployable on Streamlit Community Cloud

---

## 🏗️ Architecture

```text
User Query
   ↓
Retriever (Vector Search)
   ↓
Similarity Filter (min_score)
   ↓
Keyword Filter
   ↓
Reranker (LLM-based)
   ↓
ContextChatEngine
   ↓
Answer + Sources + Scores
```

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LlamaIndex
* OpenAI API (GPT + Vision)
* Vector Store (local)

---

## 📦 Installation (Local)

### 1. Clone the repository

```bash
git clone https://github.com/luisrvq/docs_fexplo.git
cd YOUR_REPO
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Run the app

```bash
streamlit run mainf.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push your code to GitHub
2. Go to: https://streamlit.io/cloud
3. Click **"New app"**
4. Select your repo and branch
5. Set the main file:

   ```
   mainf.py
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
├── app/
├── data/                # Uploaded documents
├── mainf.py             # Streamlit UI
├── rag_pipelinef.py     # RAG logic
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚠️ Notes

* “Open Folder” works only locally (desktop environment)
* In cloud, files are available via **Download button**
* Large files may impact performance

---

## 🔮 Future Improvements

* 📊 Visual score indicators (bars/colors)
* 🧾 Highlight matched text in snippets
* 🗂️ Multi-document filtering
* 🧠 Answer confidence scoring
* 🧹 Bulk document management

---

## 👤 Author

Luis Solis Vasquez

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
