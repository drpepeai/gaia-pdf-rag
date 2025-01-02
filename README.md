# Gaia RAG: PDF Question-Answering with Gaia and Qdrant

Gaia PDF RAG is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents using a local Gaia node and Qdrant vector database. It combines the power of local LLMs with efficient vector search to provide accurate, context-aware answers.

## Features

- 📑 PDF document processing and chunking
- 🔍 Semantic search using Qdrant vector database
- 🤖 Local LLM integration through Gaia node
- ↗️ Cross-encoder reranking for improved relevance
- 💨 Streaming responses for better UX
- 🎯 Smart source citation
- ⚡ Relevance filtering to prevent hallucinations

## Prerequisites

Before running GaiaRAG, ensure you have:

1. A local Gaia node running (Check this link to learn how to run your own local LLM: [https://docs.gaianet.ai/node-guide/quick-start](https://docs.gaianet.ai/node-guide/quick-starthttps://docs.gaianet.ai/node-guide/quick-start))
2. Qdrant server running
3. Python 3.8+
4. Required system libraries for PDF processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/harishkotra/gaia-pdf-rag.git
cd gaiarag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setting Up Components

### 1. Gaia Node

Start your local Gaia node:
```bash
gaianet init
gaianet start
```

### 2. Qdrant Server

Start Qdrant using Docker:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

## Running the Application

1. Make sure both Gaia node and Qdrant are running

2. Start the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`

## Usage

1. Upload a PDF document using the sidebar
2. Click "Process Document" to index it
3. Ask questions in the main input field
4. View answers and relevant source documents

## Configuration

You can modify the following parameters in `app.py`:

- `GAIA_NODE_URL`: URL of your local Gaia node
- `QDRANT_HOST`: Qdrant server host
- `QDRANT_PORT`: Qdrant server port
- `VECTOR_SIZE`: Embedding dimension size
- `COLLECTION_NAME`: Name for vector database collection

## Project Structure

```
gaia-pdf-rag/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Gitignore file
├── README.md           # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Credits

Inspired by [this example](https://github.com/yankeexe/llm-rag-with-reranker-demo).