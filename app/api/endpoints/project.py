# 프로젝트 관리 API
from fastapi import APIRouter

# 라우터 생성 (main.py에서 이 변수를 찾습니다!)
router = APIRouter()


# 임시 테스트용 엔드포인트
@router.get("/")
async def get_project_status():
    return {"message": "프로젝트 API가 정상적으로 연결되었습니다!"}


@router.post("/create")
async def create_project():
    # 나중에 프로젝트 생성 로직이 들어갈 자리입니다.
    return {"message": "프로젝트가 생성되었습니다."}
