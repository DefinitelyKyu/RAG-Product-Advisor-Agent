import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("LLM_API_KEY")
)


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the following user query to be more specific and searchable for IT products.
Keep it concise. Return only the rewritten query, nothing else.

Original query: {query}
Rewritten query:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def budget_filter(docs: List[Document], budget: float) -> List[Document]:
    return [
        doc for doc in docs
        if doc.metadata.get("price_thb", float("inf")) <= budget
    ]


def extract_budget(query: str) -> float | None:
    import re
    # หาตัวเลขที่ตามด้วย THB, บาท, k, K
    patterns = [
        r"(\d[\d,]*)\s*(?:thb|baht|บาท)",
        r"(\d+)\s*k\b",
        r"under\s+(\d[\d,]*)",
        r"budget\s+(?:of\s+)?(\d[\d,]*)",
    ]
    query_lower = query.lower()
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            num = match.group(1).replace(",", "")
            value = float(num)
            if value < 10000:  # ถ้าเป็น k เช่น 150k
                value *= 1000
            return value
    return None


def compare_specs(docs: List[Document]) -> str:
    if not docs:
        return "No products to compare."

    lines = ["| Spec | " + " | ".join(doc.metadata.get("name", "?") for doc in docs) + " |"]
    lines.append("|---" * (len(docs) + 1) + "|")

    fields = ["category", "brand", "price_thb"]
    labels = {"category": "Category", "brand": "Brand", "price_thb": "Price (THB)"}

    for field in fields:
        row = f"| {labels[field]} |"
        for doc in docs:
            row += f" {doc.metadata.get(field, 'N/A')} |"
        lines.append(row)

    return "\n".join(lines)