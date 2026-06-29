from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        print("⏳ Loading reranker model...")
        _model = CrossEncoder(MODEL_NAME)
        print("✅ Reranker loaded")
    return _model


def rerank(query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
    if not docs:
        return []

    model = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    scored_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in scored_docs[:top_k]]


if __name__ == "__main__":
    from retrieval.hybrid_search import hybrid_search
    query = "server for ML workload under 200000 THB"
    candidates = hybrid_search(query, k=10)
    results = rerank(query, candidates, top_k=3)
    for doc in results:
        print(f"- {doc.metadata['name']} ({doc.metadata['price_thb']} THB)")