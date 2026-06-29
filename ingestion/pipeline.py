import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
COLLECTION_NAME = "it_products"


def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    required_columns = {
        "product_id", "name", "category",
        "brand", "price_thb", "specs", "description"
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"✅ Loaded {len(df)} products from {filepath}")
    return df


def format_document(row: pd.Series) -> Document:
    content = f"""
Product: {row['name']}
Brand: {row['brand']}
Category: {row['category']}
Price: {row['price_thb']} THB
Specs: {row['specs']}
Description: {row['description']}
    """.strip()

    metadata = {
        "product_id": str(row["product_id"]),
        "name": row["name"],
        "brand": row["brand"],
        "category": row["category"],
        "price_thb": float(row["price_thb"]),
    }

    return Document(page_content=content, metadata=metadata)


def ingest(filepath: str) -> int:
    df = load_csv(filepath)
    docs = [format_document(row) for _, row in df.iterrows()]

    print("⏳ Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embedding model loaded")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    vectorstore.add_documents(docs)
    print(f"✅ Ingested {len(docs)} products into ChromaDB")
    print(f"📁 Stored at: {CHROMA_PERSIST_DIR}")
    return len(docs)


if __name__ == "__main__":
    ingest("data/sample_catalog.csv")