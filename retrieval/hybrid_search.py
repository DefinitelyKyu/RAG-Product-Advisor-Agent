import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
COLLECTION_NAME = "it_products"


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def vector_search(query: str, k: int = 10) -> List[Document]:
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=k)


def bm25_search(query: str, docs: List[Document], k: int = 10) -> List[Document]:
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top_indices]


def hybrid_search(query: str, k: int = 5) -> List[Document]:
    # vector search
    vector_results = vector_search(query, k=10)

    # BM25 search บน vector results
    bm25_results = bm25_search(query, vector_results, k=k)

    # deduplicate โดยใช้ product_id
    seen = set()
    combined = []
    for doc in bm25_results + vector_results:
        pid = doc.metadata.get("product_id")
        if pid not in seen:
            seen.add(pid)
            combined.append(doc)
        if len(combined) >= k:
            break

    return combined


if __name__ == "__main__":
    results = hybrid_search("server for ML workload under 200000 THB")
    for doc in results:
        print(f"- {doc.metadata['name']} ({doc.metadata['price_thb']} THB)")