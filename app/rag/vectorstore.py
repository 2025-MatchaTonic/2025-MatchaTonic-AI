from typing import Any, Optional

from app.core.config import settings


def get_vectorstore() -> Optional[Any]:
    if not settings.OPENAI_API_KEY or not settings.PINECONE_API_KEY:
        return None

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore
    except ModuleNotFoundError:
        return None

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY
    )
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=settings.PINECONE_API_KEY,
    )
