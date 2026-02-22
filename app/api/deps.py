# 의존성 주입 (DB 세션 등)
# app/api/deps.py
"""
API 엔드포인트에서 공통으로 재사용할 의존성(Dependencies)을 정의하는 곳입니다.

주요 용도:
1. DB 세션 생성 및 반환 (get_db)
2. JWT 토큰 검증 및 현재 로그인한 유저 정보 확인 (get_current_user)
   -> Spring 서버에서 발급한 토큰을 FastAPI가 검증할 때 이 파일에 로직을 작성합니다.
"""
from typing import Generator

# def get_db() -> Generator:
#     try:
#         db = SessionLocal()
#         yield db
#     finally:
#         db.close()
