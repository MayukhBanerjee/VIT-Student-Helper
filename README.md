# 🎓 VIT Campus RAG Assistant

A **Retrieval-Augmented Generation (RAG)** based AI assistant that answers student queries about VIT Vellore using a custom knowledge base.

---

## 🚀 Features

* 📄 Upload and process knowledge base (PDF)
* 🧠 Semantic search using embeddings
* ⚡ Fast retrieval with Qdrant Vector DB
* 🤖 Answer generation using Gemini LLM
* 🌐 FastAPI backend with REST API
* 💬 Dark-themed chat UI with markdown support

---

## 🧠 Tech Stack

* **LLM & Embeddings:** Gemini (Google Generative AI)
* **Framework:** LangChain
* **Vector DB:** Qdrant
* **Backend:** FastAPI
* **Frontend:** HTML + JS (Markdown-enabled UI)
* **Environment:** uv (Python package manager)

---

## 📂 Project Structure

```
backend/
│
├── main.py          # RAG pipeline (embedding + retrieval + generation)
├── api.py           # FastAPI server
├── KB.pdf           # Knowledge base
├── .env             # API keys
│
frontend/
└── index.html       # Chat UI
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone <your-repo-url>
cd backend
```

---

### 2. Install Dependencies (uv)

```
uv add fastapi uvicorn langchain langchain-community langchain-google-genai qdrant-client python-dotenv
uv sync
```

---

### 3. Setup Environment Variables

Create a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key   # optional (only for cloud)
```

---

### 4. Enable Required API

In Google Cloud Console:

* Enable **Generative Language API**
* Ensure API key is unrestricted OR allowed for this API

---

### 5. Run Backend

```
uv run uvicorn api:app --reload
```

API runs at:

```
http://127.0.0.1:8000
```

Swagger docs:

```
http://127.0.0.1:8000/docs
```

---

### 6. Run Frontend

```
python -m http.server 5500
```

Open:

```
http://localhost:5500/index.html
```

---

## 🔁 How It Works

1. PDF is loaded and split into chunks
2. Chunks are converted into embeddings using Gemini
3. Stored in Qdrant vector database
4. User query → embedded → similarity search
5. Relevant chunks → passed to LLM
6. Final answer generated and returned

---

## 📌 API Endpoint

### POST `/ask`

Request:

```
{
  "query": "How is life in VIT?"
}
```

Response:

```
{
  "answer": "..."
}
```



---

## 👨‍💻 Author

Mayukh Banerjee

---

## ⭐ If you found this useful, give it a star!
