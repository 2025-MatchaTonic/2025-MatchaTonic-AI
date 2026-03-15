import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import chat, project, template
from app.api.schemas.template import NotionTemplatePayload
from app.core.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="MatchaTonic AI PM")

cors_allow_origins = settings.AI_CORS_ALLOW_ORIGINS
cors_allow_credentials = "*" not in cors_allow_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/ai/chat", tags=["AI Chat"])
app.include_router(template.router, prefix="/ai/template", tags=["AI Template"])
app.include_router(project.router, prefix="/ai/project", tags=["Project"])
app.add_api_route(
    "/ai/generate",
    template.generate_template_for_spring,
    methods=["POST"],
    response_model=NotionTemplatePayload,
    tags=["AI Template"],
)


@app.on_event("startup")
async def validate_runtime_configuration():
    if settings.REQUIRE_OPENAI_API_KEY and not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to start the AI server.")

    if settings.REQUIRE_PINECONE_FOR_RAG and not settings.PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is required when REQUIRE_PINECONE_FOR_RAG=true."
        )

    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is missing. LLM-backed endpoints will fail.")

    if not settings.PINECONE_API_KEY:
        logger.warning("PINECONE_API_KEY is missing. RAG will run without Pinecone context.")

    logger.info(
        "AI server startup validated. model=%s cors_origins=%s",
        settings.OPENAI_MODEL,
        ",".join(cors_allow_origins),
    )


@app.get("/")
async def root():
    return {"message": "AI PM Server is running!"}
