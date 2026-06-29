import os
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank
from agent.tools import rewrite_query, budget_filter, extract_budget, compare_specs
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("LLM_API_KEY")
)

SYSTEM_PROMPT = """You are an IT procurement advisor for a technology distributor.
Recommend suitable IT products based on the user's requirements and budget.
Always cite product names and prices. Be concise and professional."""


class AgentState(TypedDict):
    query: str
    rewritten_query: str
    retrieved_docs: List[Document]
    filtered_docs: List[Document]
    comparison_table: str
    answer: str
    sources: List[dict]


def node_rewrite(state: AgentState) -> AgentState:
    rewritten = rewrite_query(state["query"])
    print(f"🔄 Rewritten query: {rewritten}")
    return {**state, "rewritten_query": rewritten}


def node_retrieve(state: AgentState) -> AgentState:
    query = state["rewritten_query"] or state["query"]
    candidates = hybrid_search(query, k=10)
    top_docs = rerank(query, candidates, top_k=5)
    return {**state, "retrieved_docs": top_docs}


def node_filter(state: AgentState) -> AgentState:
    budget = extract_budget(state["query"])
    docs = state["retrieved_docs"]
    if budget:
        print(f"💰 Budget detected: {budget} THB")
        docs = budget_filter(docs, budget)
    return {**state, "filtered_docs": docs}


def node_compare(state: AgentState) -> AgentState:
    query_lower = state["query"].lower()
    is_comparison = any(word in query_lower for word in ["compare", "vs", "versus", "difference", "between"])
    if is_comparison:
        table = compare_specs(state["filtered_docs"])
    else:
        table = ""
    return {**state, "comparison_table": table}


def node_generate(state: AgentState) -> AgentState:
    docs = state["filtered_docs"] or state["retrieved_docs"]
    context = "\n\n".join([doc.page_content for doc in docs])

    comparison = state.get("comparison_table", "")
    if comparison:
        context += f"\n\nComparison Table:\n{comparison}"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}")
    ]

    response = llm.invoke(messages)

    sources = [
        {
            "product_id": doc.metadata.get("product_id"),
            "name": doc.metadata.get("name"),
            "price_thb": doc.metadata.get("price_thb"),
            "category": doc.metadata.get("category"),
        }
        for doc in docs
    ]

    return {**state, "answer": response.content, "sources": sources}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("rewrite", node_rewrite)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("filter", node_filter)
    graph.add_node("compare", node_compare)
    graph.add_node("generate", node_generate)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "filter")
    graph.add_edge("filter", "compare")
    graph.add_edge("compare", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


agent = build_graph()


if __name__ == "__main__":
    result = agent.invoke({
        "query": "recommend a server for ML workload under 200000 THB",
        "rewritten_query": "",
        "retrieved_docs": [],
        "filtered_docs": [],
        "comparison_table": "",
        "answer": "",
        "sources": []
    })
    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    for s in result["sources"]:
        print(f"- {s['name']} ({s['price_thb']} THB)")