# app/db/models.py
"""
FastAPI 전용 데이터베이스 테이블(Model)을 정의하는 곳입니다.
(예: AI 토큰 사용량 로깅 테이블, LangGraph 체크포인트(대화 기록) 저장 테이블 등)

추후 확장을 위해 자리를 비워둡니다.
"""
# from sqlalchemy import Column, Integer, String
# from .base import Base

# class AILog(Base):
#     __tablename__ = "ai_logs"
#     id = Column(Integer, primary_key=True, index=True)
#     project_id = Column(String, index=True)
#     tokens_used = Column(Integer)
