from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import chat, project, template

app = FastAPI(title="MatchaTonic AI PM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/ai/chat", tags=["AI Chat"])
app.include_router(template.router, prefix="/ai/template", tags=["AI Template"])
app.include_router(project.router, prefix="/ai/project", tags=["Project"])


@app.get("/")
async def root():
    return {"message": "AI PM Server is running!"}


# 서버 실행 명령어:
# uvicorn main:app --reload --port 8000
