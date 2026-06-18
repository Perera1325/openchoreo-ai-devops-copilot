from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    Returns the HuggingFace embeddings model (all-MiniLM-L6-v2).
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
