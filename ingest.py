import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Configuration
DATA_PATH = "./data"
DB_PATH = "./chroma_db"

def ingest_documents():
    print("Starting Ingestion...")

    # 2. Load Documents with Production-Grade Metadata
    documents = []
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} directory not found.")
        return

    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, filename)
            
            # Load the file
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            
            # Add Metadata (Crucial for filtering later)
            # This tells the system: "This chunk came from the Battlecard"
            for doc in docs:
                doc.metadata["source"] = filename
                if "price" in filename.lower() or "specs" in filename.lower():
                    doc.metadata["type"] = "technical"
                else:
                    doc.metadata["type"] = "general"
            
            documents.extend(docs)
            print(f"   - Loaded: {filename}")

    # 3. Chunking Strategy
    # We use 500 chars with 50 overlap. 
    # Small chunks = precise retrieval for specific sales questions.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} text chunks.")

    # 4. Embeddings (The Open Source Powerhouse)
    # This runs LOCALLY. No API cost.
    print("Generating Embeddings (HuggingFace - all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Store in ChromaDB
    print("Saving to Vector Database...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Ingestion Complete! Database saved to ./chroma_db")

if __name__ == "__main__":
    ingest_documents()