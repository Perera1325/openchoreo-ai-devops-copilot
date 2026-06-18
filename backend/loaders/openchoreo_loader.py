import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain_core.documents import Document

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "documents", "openchoreo")

def load_openchoreo_docs() -> List[Document]:
    docs = []
    
    loaders = {
        "**/*.md": TextLoader,
        "**/*.txt": TextLoader,
        "**/*.pdf": PyPDFLoader,
        "**/*.html": UnstructuredHTMLLoader,
    }
    
    if not os.path.exists(DOCS_DIR):
        return docs

    for glob_pattern, loader_cls in loaders.items():
        loader = DirectoryLoader(DOCS_DIR, glob=glob_pattern, loader_cls=loader_cls)
        try:
            loaded_docs = loader.load()
            for doc in loaded_docs:
                # Add default metadata if missing
                source = doc.metadata.get("source", "Unknown")
                doc.metadata["source"] = source
                if "title" not in doc.metadata:
                    doc.metadata["title"] = os.path.basename(source)
                if "url" not in doc.metadata:
                    doc.metadata["url"] = f"file://{os.path.abspath(source)}"
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading OpenChoreo documents matching {glob_pattern}: {e}")
            
    return docs
