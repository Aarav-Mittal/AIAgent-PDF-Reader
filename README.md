# 🤖 Autonomous AI Agent — Math Engine & PDF RAG Assistant

An autonomous, multi-modal AI Agent system built using **Model Context Protocol (MCP)**, **LangChain**, **Groq (Llama 3.1 8B)**, **ChromaDB**, and **Streamlit**. 

This system guarantees zero arithmetic hallucinations by routing mathematical expressions to a safe, deterministic FastMCP execution server while serving localized Retrieval-Augmented Generation (RAG) over uploaded PDF documents using local HuggingFace embeddings.

---

## 🌟 Key Features

- **Model Context Protocol (MCP) Architecture**: Decoupled tool microservices server (`mcp_server.py`) running over standardized HTTP transport (`streamable-http`).
- **Zero-Hallucination Math Engine**: Evaluates complex mathematical expressions, scientific formulas, trigonometric functions, powers, and factorials using safe AST evaluation rather than LLM token guessing.
- **Local PDF RAG Pipeline**:
  - PDF document loading with `PyPDFLoader`.
  - Smart section-aware text splitting via `RecursiveCharacterTextSplitter`.
  - High-performance local vector embeddings with HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`).
  - Persistent local vector store backed by `ChromaDB`.
- **Deep Topic Analysis & Study Tools**: Dedicated tab for topic summaries, bulleted study notes, and open-ended subjective practice questions.
- **Transparent Execution Audit Log**: Real-time visual cards displaying every tool invoked by the agent, including input parameters and output observations.
- **Resilient Fallback Handling**: Automatic exception interception for API rate limits (`429`) and function-calling parsing errors.

---

## 📐 System Architecture

```
                                 ┌─────────────────────────────────┐
                                 │   Streamlit Web Interface       │
                                 │          (app.py)               │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │  LangChain Agent Executor       │
                                 │     (Groq Llama 3.1 8B)         │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼ (Streamable HTTP / JSON-RPC)
                                 ┌─────────────────────────────────┐
                                 │      FastMCP Server             │
                                 │       (port 8050)               │
                                 └───────┬────────────────┬────────┘
                                         │                │
                                         ▼                ▼
                         ┌──────────────────┐    ┌──────────────────┐
                         │   Math Engine    │    │ Local ChromaDB   │
                         │ (Deterministic)  │    │  (Vector Store)  │
                         └──────────────────┘    └──────────────────┘
```

---

## 📁 Repository Structure

```
├── app.py                  # Streamlit frontend & agent execution coordinator
├── mcp_server.py           # FastMCP tools server exposing Math & RAG tools over HTTP
├── client.py               # MCP client helper / test script
├── main.py                 # Alternative Streamlit runner script
├── run_ui.bat              # One-click Windows batch script to launch server & UI
├── pyproject.toml          # Project dependencies and configuration (uv compatible)
├── temp_uploads/           # Storage directory for uploaded PDF documents
├── chroma_db/              # Persistent Chroma vector store database directory
└── README.md               # Project documentation
```

---

## 🛠️ Tech Stack

- **Frontend UI**: Streamlit (with custom Glassmorphic styling)
- **Agent Orchestration**: LangChain Classic, LangChain Core, LangChain Groq
- **Protocol**: Model Context Protocol (FastMCP + `langchain-mcp-adapters`)
- **LLM Engine**: Groq (`llama-3.1-8b-instant`)
- **Vector Database**: ChromaDB
- **Embedding Model**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **PDF Parser**: PyPDF Loader

---

## ⚡ Quickstart & Installation

### Prerequisites
- Python 3.11+ or 3.13 (recommended with `uv`)
- Groq API Key (Sign up at [console.groq.com](https://console.groq.com))
- OpenWeatherMap API Key (optional, for weather tool)

### 1. Environment Setup
Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
OWM_API_KEY=your_openweathermap_api_key_here
```

### 2. Install Dependencies
Using `uv` (recommended):
```bash
uv sync
```

Or using standard `pip`:
```bash
pip install -r pyproject.toml
```

### 3. Launching the Application

**Option A: One-Click Launch (Windows Batch)**
Double-click or run:
```cmd
run_ui.bat
```

**Option B: Manual Command Line Launch**
1. Start the FastMCP Server in terminal 1:
   ```bash
   uv run python mcp_server.py
   ```
2. Start the Streamlit UI in terminal 2:
   ```bash
   uv run streamlit run app.py
   ```

---

## 🚀 Usage Guide

1. **Verify Server Status**: Ensure the sidebar shows **FastMCP Server Online** (`http://localhost:8050/mcp`).
2. **Mathematical Calculations**: Ask any formula or math question (e.g., `Calculate sin(radians(45)) + sqrt(144)`). Check the expander to view tool execution logs.
3. **PDF Document RAG**:
   - Upload your PDF file in the sidebar.
   - Click **⚡ Index PDF in ChromaDB**.
   - Ask questions related to the PDF content in the chat interface.
   - Switch to the **📚 RAG Topic Analysis** tab for auto-generated summaries, study notes, and quiz questions.

---
