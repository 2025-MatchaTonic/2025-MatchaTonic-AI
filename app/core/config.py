# 환경변수 로드 (Pydantic Settings)
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-pm-knowledge")


settings = Settings()
