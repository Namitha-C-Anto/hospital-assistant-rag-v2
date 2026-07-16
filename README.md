# 🏥 Hospital Assistant RAG v2

An AI-powered conversational assistant that answers hospital policy and patient service questions using **Retrieval-Augmented Generation (RAG)**.

Unlike a standard chatbot, responses are grounded in hospital policy documents through **hybrid retrieval**, **reranking**, and **source attribution**, reducing hallucinations and improving answer quality.

---

## 🚀 Live Demo

🌐 **Streamlit App:** https://hospital-assistant-rag-v2.streamlit.app/

📂 **Repository:** https://github.com/Namitha-C-Anto/hospital-assistant-rag-v2

---
## 🎥 Demo

https://github.com/user-attachments/assets/dd7e4bf1-449b-48c5-892f-ce10451be780

---

# Highlights

🚀 End-to-end conversational RAG application
🔍 Hybrid retrieval using FAISS + BM25 + Reciprocal Rank Fusion (RRF)
🎯 Cross-encoder reranking for improved retrieval quality
📊 Integrated RAGAS evaluation framework
💬 Multi-chat Streamlit interface with source attribution
☁️ Deployed on Streamlit Community Cloud

# Features

### 💬 Conversational Chat Interface

- Multi-chat support
- Persistent conversation history
- Create, switch, rename, and delete chats
- Follow-up question support

---

### 🔍 Hybrid Retrieval

Combines multiple retrieval techniques for improved recall.

- FAISS Vector Search
- BM25 Keyword Search
- Reciprocal Rank Fusion (RRF)
- Duplicate removal

---

### 🎯 Cross-Encoder Reranking

Retrieved documents are reranked using

**BAAI/bge-reranker-base**

to improve context relevance before sending them to the LLM.

---

### 🤖 AI Answer Generation

Responses are generated using OpenAI GPT models while grounding every answer in retrieved hospital documents.

---

### 📚 Source Transparency

Every answer includes:

- Source PDF
- Page Number
- Retrieved Context

making the assistant easier to verify and debug.

---

# Architecture

```
                    User Question
                          │
                          ▼
                 Conversation History
                          │
                          ▼
                  Query Processing
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
     FAISS Search                   BM25 Search
          │                               │
          └───────────────┬───────────────┘
                          ▼
              Reciprocal Rank Fusion
                          │
                          ▼
                 Duplicate Removal
                          │
                          ▼
              Cross-Encoder Reranker
                          │
                          ▼
                Top Ranked Documents
                          │
                          ▼
                  OpenAI GPT Model
                          │
                          ▼
                 Answer + Source Chunks
```

🏠 Home screen
<img width="1158" height="566" alt="welcome_screen" src="https://github.com/user-attachments/assets/33b523b0-1e22-4bbc-afda-2af4603965c3" />

💬 Chat conversation
<img width="1165" height="570" alt="chat_interface" src="https://github.com/user-attachments/assets/2449d37b-3d92-45b0-880a-8d4b06d1df0c" />

📚 Source panel
<img width="1148" height="585" alt="source" src="https://github.com/user-attachments/assets/c0e48870-2446-4974-ad3b-6822a8639ed4" />

---

# Tech Stack

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| UI | Streamlit |
| LLM | OpenAI GPT |
| Framework | LangChain |
| Embeddings | sentence-transformers |
| Vector Database | FAISS |
| Keyword Search | BM25 |
| Hybrid Search | Ensemble + RRF |
| Reranker | BAAI/bge-reranker-base |
| Evaluation | RAGAS |
| PDF Processing | PyMuPDF |
| Version Control | Git & GitHub |

---

# Project Structure

```
Hospital_Assistant_v2.0
│
├── app.py
├── config.py
│
├── docs/
├── db/
│
├── rag/
│   ├── builder.py
│   ├── initializer.py
│   ├── pipeline.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── vectorstore.py
│   └── ...
│
├── memory/
│
├── ui/
│
├── prompts/
│
├── llm/
│
├── evaluate/
│
└── utils/
```

---

# Retrieval Pipeline

1. User submits a question.
2. Conversation history is formatted.
3. FAISS retrieves semantically similar chunks.
4. BM25 retrieves keyword-matching chunks.
5. Results are merged using Reciprocal Rank Fusion.
6. Duplicate chunks are removed.
7. Cross-Encoder reranks the retrieved documents.
8. Top-ranked contexts are passed to the LLM.
9. Final answer is generated with supporting sources.

---
# 📊 RAG Evaluation

This project includes an automated evaluation pipeline built with **RAGAS** to measure retrieval and generation quality across multiple retrieval strategies.

## Evaluation Metrics

The following metrics were used to evaluate the system:

| Metric | Description |
|---------|-------------|
| **Faithfulness** | Measures whether the generated answer is supported by the retrieved context. |
| **Answer Relevancy** | Measures how well the answer addresses the user's question. |
| **Context Precision** | Measures the relevance of the retrieved documents. |
| **Context Recall** | Measures whether the retrieval pipeline returned all required information. |

---

## Evaluation Dataset

- **Dataset:** Hospital Policy Dataset (dataset_v2)
- **Test Questions:** 22
- **Embedding Model:** sentence-transformers/all-mpnet-base-v2
- **LLM:** GPT-4o-mini
- **Judge Model:** GPT-5.4-mini

---

## Retrieval Strategy Comparison

The RAG pipeline was evaluated using multiple retrieval configurations.

| Retrieval Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|--------------------|-------------:|-----------------:|------------------:|---------------:|
| FAISS | 0.961 | 0.914 | 0.555 | 0.955 |
| Hybrid (FAISS + BM25) | **1.000** | **0.968** | **1.000** | **1.000** |
| Hybrid (No Reranker) | 0.875 | 0.968 | 0.833 | 1.000 |

The experiments demonstrate that combining semantic search (FAISS) with keyword search (BM25) and Cross-Encoder reranking significantly improved retrieval quality compared to vector search alone.

---

## Evaluation Pipeline

The evaluation framework automatically generates:

- ✅ Per-question RAGAS metrics
- ✅ Overall evaluation summaries
- ✅ JSON experiment metadata
- ✅ CSV reports
- ✅ Retrieval latency metrics
- ✅ Generation latency metrics
- ✅ Token usage statistics
- ✅ Experiment comparison reports

---

## Evaluation Artifacts

The repository includes detailed evaluation outputs:

```
evaluate/
└── evaluation_results/
    ├── datasets/
    ├── ragas_results/
```

These artifacts make it possible to compare different retrieval strategies, embedding models, chunking configurations, and reranking approaches.

---


# Running Locally

Clone the repository

```bash
git clone https://github.com/Namitha-C-Anto/hospital-assistant-rag-v2.git
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate

```bash
source .venv/bin/activate
```

or

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env`

```text
OPENAI_API_KEY=your_key
```

Run

```bash
streamlit run app.py
```

---

# Future Improvements

- Qdrant Vector Database
- Metadata Filtering
- FastAPI Backend
- Docker Deployment
- Authentication
- Agentic AI Workflows
- Cloud Deployment
- Observability with LangSmith

---

# Key Learnings

This project explores several practical RAG engineering techniques including:

- Hybrid Retrieval
- Reciprocal Rank Fusion
- Cross-Encoder Reranking
- Retrieval Evaluation
- Modular Pipeline Design
- Conversation Memory
- Streamlit UI Development
- Production-style Project Organization

---
## Roadmap

### Version 1
- Basic RAG pipeline
- FAISS retrieval
- Streamlit interface

### Version 2 (Current)
- Hybrid retrieval (FAISS + BM25)
- Reciprocal Rank Fusion
- Optional Cross-Encoder reranking
- Source attribution
- RAGAS evaluation
- Modular architecture

### Version 3 (Planned)
- Qdrant vector database
- Metadata filtering
- Self-query retriever
- FastAPI backend
- Authentication
- Docker deployment
---
Note: This project uses the OpenAI API for answer generation. If the live demo is temporarily unavailable due to API quota limits, the repository includes evaluation results, screenshots, and deployment instructions. A demo video is also provided to showcase the application's functionality.
---
# Author

**Namitha C Anto**

AI Engineer | Generative AI | RAG | Agentic AI

GitHub:
https://github.com/Namitha-C-Anto

LinkedIn:
https://www.linkedin.com/in/namitha-c-anto-79442b103

