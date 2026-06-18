from typing import List
from fastapi import APIRouter
from pydantic import BaseModel, Field
from rag.rag_pipeline import generate_rag_response

router = APIRouter(
    prefix="/api",
    tags=["chat"]
)

class ChatRequest(BaseModel):
    question: str = Field(..., description="The user's question regarding DevOps or WSO2/OpenChoreo documentation")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The generated response from the AI assistant")
    sources: List[str] = Field(default_factory=list, description="List of sources used to generate the answer")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle chat requests to the AI DevOps Copilot.
    Uses RAG with local Ollama (gemma3:4b) and ChromaDB.
    """
    answer, sources = generate_rag_response(request.question)
    return ChatResponse(
        answer=answer,
        sources=sources
    )
