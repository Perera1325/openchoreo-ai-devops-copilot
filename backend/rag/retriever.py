import os
from typing import List
from loaders.wso2_loader import load_wso2_docs
from loaders.openchoreo_loader import load_openchoreo_docs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from rag.embeddings import get_embeddings

VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "vector_db")

def load_all_documents() -> List:
    docs = []
    docs.extend(load_wso2_docs())
    docs.extend(load_openchoreo_docs())
    return docs

def build_or_get_retriever():
    embeddings = get_embeddings()
    
    # Check if vector DB already exists, otherwise build it
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    else:
        docs = load_all_documents()
        if not docs:
            # Create an empty vector store if no documents exist yet
            vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=VECTOR_DB_DIR)
            
    return vectorstore.as_retriever(search_kwargs={"k": 5})
