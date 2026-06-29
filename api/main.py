import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from ingestion.pipeline import ingest
import shutil
import tempfile
from agent.graph import agent


load_dotenv()

app = FastAPI(title="Procure-AI", version="0.1.0")

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL"),
    google_api_key=os.getenv("LLM_API_KEY")
)

SYSTEM_PROMPT = """You are an IT procurement advisor for a technology distributor.
Your job is to recommend suitable IT products based on the user's requirements and budget.
Always cite which products you are referring to by name and price.
Be concise, helpful, and professional."""


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = agent.invoke({
        "query": request.query,
        "rewritten_query": "",
        "retrieved_docs": [],
        "filtered_docs": [],
        "comparison_table": "",
        "answer": "",
        "sources": []
    })
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

@app.post("/ingest")
def ingest_catalog(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    count = ingest(tmp_path)
    return {"message": f"Ingested {count} products successfully"}


@app.get("/products")
def list_products():
    from retrieval.hybrid_search import get_vectorstore
    vectorstore = get_vectorstore()
    results = vectorstore.get()
    return {"count": len(results["ids"]), "products": results["metadatas"]}