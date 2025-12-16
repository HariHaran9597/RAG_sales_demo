import os
import argparse
from dotenv import load_dotenv

# LangChain Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables (API Key)
load_dotenv()

# 2. Configuration
DB_PATH = "./chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in .env file")
    exit(1)

def query_rag_system(question):
    print(f"\nQuerying: '{question}'")
    print("-" * 40)

    # 3. Load the Local Vector DB
    # Must use the SAME embedding model as ingestion!
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    
    # 4. Setup the Retriever (Fetch top 3 relevant chunks)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 5. Initialize Groq (The High-Speed LLM)
    llm = ChatGroq(
        temperature=0, 
        model_name="moonshotai/kimi-k2-instruct-0905", 
        groq_api_key=GROQ_API_KEY
    )

    # 6. The "Sales Expert" Prompt
    # This prompt forces the model to stick to the facts.
    template = """You are an elite Sales Engineer for VelocityAI.
    Use the provided context to answer the user's question.
    
    Rules:
    1. If the answer is not in the context, say "I don't have that info."
    2. Be concise and professional.
    3. If mentioning a competitor, explain how we win.

    Context:
    {context}

    Question:
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # 7. Build the Chain (The RAG Pipeline)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 8. Run it!
    response = rag_chain.invoke(question)
    print("AI Response:")
    print(response)
    print("-" * 40)

    # Optional: Show what documents were used (Transparency)
    print("Sources Used:")
    docs = retriever.invoke(question)
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        print(f" - {source}")

if __name__ == "__main__":
    # Allow running from command line: python query.py "My question"
    import sys
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        query_rag_system(user_input)
    else:
        # Default test question
        query_rag_system("How is VelocityAI better than LegacyCRM?")