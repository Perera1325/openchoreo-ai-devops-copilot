from typing import Tuple, List
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from rag.retriever import build_or_get_retriever
from rag.prompt import rag_prompt

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_or_get_retriever()
    return _retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_rag_response(question: str) -> Tuple[str, List[str]]:
    """
    Generates a response using the RAG pipeline.
    Returns a tuple of (answer, list_of_sources).
    """
    llm = ChatOllama(model="gemma3:4b")
    retriever = get_retriever()
    
    try:
        docs = retriever.invoke(question)
        context = format_docs(docs)
        
        chain = rag_prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "input": question})
        
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
        return answer, sources
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return f"Error generating response: {e}", []
