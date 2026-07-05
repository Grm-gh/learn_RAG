# 🤖 RAG AI Assistant using LangChain, ChromaDB & Mistral AI

A modern **Retrieval-Augmented Generation (RAG) chatbot** built using **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and **Mistral AI** with an interactive **Streamlit UI**.

This project demonstrates how Large Language Models can answer user queries using external knowledge stored in a vector database, enabling accurate and context-aware responses.

---

## 🚀 Features

- 🔍 Retrieval-Augmented Generation (RAG)
- 🧠 Mistral AI LLM integration
- 📚 ChromaDB vector database
- 🔗 LangChain framework
- 🤗 HuggingFace sentence embeddings
- 🎯 Maximum Marginal Relevance (MMR) retrieval
- 💬 Interactive Streamlit chat interface
- 📖 Retrieved context visualization
- ⚡ Fast semantic search
- 🎨 Modern glassmorphism UI

---

## 🏗️ Architecture

```text
                User Query
                     │
                     ▼
            HuggingFace Embedding
                     │
                     ▼
              Chroma Vector DB
                     │
                     ▼
             Similarity Retrieval
                     │
                     ▼
              Context Generation
                     │
                     ▼
                Mistral LLM
                     │
                     ▼
               Final Response
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend |
| LangChain | RAG Framework |
| ChromaDB | Vector Database |
| HuggingFace | Text Embeddings |
| Mistral AI | Large Language Model |
| Streamlit | User Interface |
| dotenv | Environment Variables |

---

## 📁 Project Structure

```text
raglearn/
│
├── .venv/
├── chroma-db/
├── document loader/
├── vector store/
│   └── db.py
├── .env
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone <repository-url>
cd raglearn
```

### Create virtual environment

```bash
python -m venv .venv
```

### Activate environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux/Mac

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

---

## 📚 Creating Vector Database

Run:

```bash
python "vector store/db.py"
```

This will:

- Create embeddings using HuggingFace
- Store vectors in ChromaDB
- Persist the database locally

---

## 🚀 Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🔍 Retrieval Strategy

This project uses **Maximum Marginal Relevance (MMR)** retrieval:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| k | Number of documents returned |
| fetch_k | Number of candidate documents |
| lambda_mult | Diversity vs relevance balance |

---

## 🧠 Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

- Lightweight embedding model
- 384-dimensional vectors
- Optimized for semantic similarity search

---

## 🤖 LLM Used

```text
mistral-small-2603
```

Features:

- Fast inference
- Strong reasoning capabilities
- Context-aware responses
- Production-grade performance

---

## 📷 Application Features

### Chat Interface

- Interactive conversation
- Persistent chat history
- Real-time responses

### Retrieved Context Viewer

- Displays retrieved documents
- Shows source information
- Helps visualize RAG workflow

### Performance Metrics

- Retrieval latency
- Number of documents retrieved
- Response statistics

---

## 🔄 RAG Workflow

```text
User Question
      ↓
Query Embedding
      ↓
Vector Similarity Search
      ↓
Retrieve Relevant Chunks
      ↓
Build Prompt
      ↓
Send to Mistral
      ↓
Generate Response
      ↓
Display Answer
```

---

## 📌 Example Query

### User

```text
What is LangChain?
```

### Retrieved Context

```text
LangChain is a framework used for building
applications with large language models.
```

### Response

```text
LangChain is a framework used for building
applications with large language models.
```

---

## 🎯 Learning Outcomes

This project demonstrates:

- Vector embeddings
- Semantic search
- Retrieval-Augmented Generation
- Vector databases
- Prompt engineering
- Large Language Models
- LangChain framework
- Streamlit UI development

---

## 👨‍💻 Author

**Gyanu Rajmaniar**

B.Tech Computer Science & Engineering  
Birla Institute of Technology, Mesra

---

## ⭐ Future Improvements

- PDF document upload
- Multi-document RAG
- Conversation memory
- Source citation
- Hybrid search
- Re-ranking models
- Authentication
- Cloud deployment

---

## 📄 License

This project is developed for educational and learning purposes.
