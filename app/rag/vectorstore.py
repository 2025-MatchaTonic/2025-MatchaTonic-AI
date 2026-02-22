from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings


def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY
    )

    vectorstore = PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=settings.PINECONE_API_KEY,
    )
    return vectorstore
