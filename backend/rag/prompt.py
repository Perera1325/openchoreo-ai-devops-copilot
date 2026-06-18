from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an AI DevOps Copilot answering questions about WSO2 and OpenChoreo.\n"
    "Use the following pieces of retrieved context to answer the question.\n"
    "If you don't know the answer, just say that you don't know. Do not make up information.\n"
    "\n"
    "Context:\n"
    "{context}"
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
