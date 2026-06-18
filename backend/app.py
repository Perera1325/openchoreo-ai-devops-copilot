from fastapi import FastAPI
from api.chat import router as chat_router

app = FastAPI(
    title="OpenChoreo AI DevOps Copilot",
    description="AI-powered DevOps assistant using RAG for WSO2 and OpenChoreo documentation"
)

app.include_router(chat_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to OpenChoreo AI DevOps Copilot"}
