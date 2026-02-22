# app/db/session.py
"""
데이터베이스와 연결하는 엔진(Engine)과 세션(Session)을 생성하는 곳입니다.
추후 SQLite나 PostgreSQL을 FastAPI에 직접 연결할 때 사용합니다.
"""
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from app.core.config import settings

# engine = create_engine(settings.DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
