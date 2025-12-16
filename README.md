# RAG Sales Demo

A Retrieval-Augmented Generation (RAG) system designed for sales engineering teams to quickly answer questions about products, competitors, and past sales conversations using AI.

## Overview

This project combines:
- **Vector Database (ChromaDB)**: Stores embeddings of sales documents for fast retrieval
- **LLM (Groq)**: High-speed language model for intelligent responses
- **HuggingFace Embeddings**: Open-source embedding model running locally (no API costs)
- **LangChain**: Framework orchestrating the RAG pipeline

## Features

- Query sales documents with natural language questions
- Get AI-powered answers grounded in your actual sales materials
- Transparent source attribution (see which documents were used)
- Local embeddings (no external API calls for embedding generation)
- Fast responses via Groq's optimized inference

## Project Structure

```
.
├── ingest.py                 # Loads documents into the vector database
├── query.py                  # CLI tool to query the RAG system
├── requirements.txt          # Python dependencies
├── data/                     # Sales documents
│   ├── competitor_battlecard.txt
│   ├── past_call_transcript.txt
│   └── velocity_ai_specs.txt
└── chroma_db/               # Vector database (auto-created)
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Ingest Documents
```bash
python ingest.py
```
This loads all `.txt` files from the `data/` folder into ChromaDB with metadata.

## Usage

### Command Line
```bash
python query.py "Your question here"
```

Examples:
```bash
python query.py "How is VelocityAI better than LegacyCRM?"
python query.py "Is our data secure?"
python query.py "What are our pricing options?"
```

### Default Query
```bash
python query.py
```
Runs a default comparison question.

## How It Works

1. **Ingestion** (`ingest.py`):
   - Loads `.txt` files from `data/` directory
   - Splits documents into 500-character chunks with 50-character overlap
   - Generates embeddings using HuggingFace's all-MiniLM-L6-v2 model
   - Stores in ChromaDB with metadata (source, type)

2. **Query** (`query.py`):
   - Retrieves top 3 most relevant chunks from the vector database
   - Sends context + question to Groq LLM
   - Uses a "Sales Expert" prompt to ground responses in provided context
   - Returns AI response with source attribution

## Adding More Documents

1. Add `.txt` files to the `data/` folder
2. Run `python ingest.py` to re-index
3. Query with `python query.py`

## API Keys

- **Groq API Key**: Get from [console.groq.com](https://console.groq.com)
  - Free tier available with rate limits
  - Excellent for development and testing

## Technologies

- **LangChain**: RAG orchestration
- **ChromaDB**: Vector database
- **HuggingFace**: Open-source embeddings
- **Groq**: High-speed LLM inference
- **Python 3.10+**

## License

MIT
