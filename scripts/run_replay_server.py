import argparse
import importlib
import os
import sys
from types import MethodType
from pathlib import Path

import uvicorn
from langchain_openai import ChatOpenAI


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


def _build_fake_template_json() -> str:
    return (
        '{"summary_message":"승인된 collected_data 기준으로 보수적인 템플릿 초안을 만들었습니다.",'
        '"project_home":{"project_overview":"현재 확보된 정보만 반영한 초안입니다."},'
        '"planning":{"project_intro":"수집된 정보 기준으로 프로젝트 개요를 정리합니다.",'
        '"problem_definition":[{"id":1,"situation":"현재 문제 상황은 추가 확인이 필요합니다.","reason":"","limitation":""}],'
        '"solution":{"core_summary":"확정된 정보만 반영해 다음 단계를 정리합니다.","problem_solutions":[{"problem_id":1,"solution_desc":"핵심 정보부터 확정합니다."}],"features":["확정 정보 정리","누락 정보 확인","다음 단계 제안"]},'
        '"target_persona":{"name":"","age":"","job_role":"","main_activities":"","pain_points":[],"needs":[]}},'
        '"ground_rules":"확정되지 않은 정보는 추정하지 않고 추가 확인 후 반영합니다."}'
    )


def _build_fake_gather_json() -> str:
    return (
        '{"intent":"general","ai_message":"현재 입력 기준으로 확인된 내용만 반영할게요. 필요한 항목을 짧게 말해 주세요.",'
        '"updated_data":{},"is_sufficient":false}'
    )


def _fake_response_for(prompt: str, **kwargs):
    normalized = (prompt or "").lower()
    if "summary_message" in normalized and "project_home" in normalized:
        return _FakeResponse(_build_fake_template_json())
    if kwargs.get("response_format") or "updated_data" in normalized or "json only" in normalized:
        return _FakeResponse(_build_fake_gather_json())
    return _FakeResponse(
        "현재 입력 기준으로 확인된 내용만 반영할게요. 필요한 항목을 짧게 말해 주세요."
    )


def _fake_invoke_llm(*args, **kwargs):
    prompt = ""
    if len(args) >= 2:
        prompt = str(args[1] or "")
    elif args:
        prompt = str(args[0] or "")
    return _fake_response_for(prompt, **kwargs)


def _patch_model_invoke(model):
    if model is None:
        return

    def _invoke(self, prompt, **kwargs):
        return _fake_response_for(str(prompt or ""), **kwargs)

    model.invoke = MethodType(_invoke, model)


def _patch_chatopenai_class():
    def _invoke(self, prompt, **kwargs):
        return _fake_response_for(str(prompt or ""), **kwargs)

    ChatOpenAI.invoke = _invoke


def _patch_runtime(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    _patch_chatopenai_class()
    main_module = importlib.import_module("main")

    try:
        nodes = importlib.import_module("app.ai.graph.nodes")
        if hasattr(nodes, "_invoke_llm"):
            nodes._invoke_llm = _fake_invoke_llm
        if hasattr(nodes, "_fetch_rag_context"):
            nodes._fetch_rag_context = lambda *args, **kwargs: getattr(
                nodes, "RAG_EMPTY_CONTEXT", ""
            )
        if hasattr(nodes, "get_rag_context"):
            nodes.get_rag_context = lambda *args, **kwargs: getattr(
                nodes, "RAG_EMPTY_CONTEXT", ""
            )
        _patch_model_invoke(getattr(nodes, "conversation_llm", None))
        _patch_model_invoke(getattr(nodes, "structured_llm", None))
    except Exception as exc:
        print(f"[replay-server] failed to patch nodes: {exc}", file=sys.stderr)

    try:
        llm_clients = importlib.import_module("app.ai.graph.llm_clients")
        if hasattr(llm_clients, "invoke_llm"):
            llm_clients.invoke_llm = _fake_invoke_llm
        _patch_model_invoke(getattr(llm_clients, "conversation_llm", None))
        _patch_model_invoke(getattr(llm_clients, "structured_llm", None))
    except Exception as exc:
        print(f"[replay-server] failed to patch llm clients: {exc}", file=sys.stderr)

    for module_name in (
        "app.api.endpoints.template",
        "app.ai.services.template_generation",
    ):
        try:
            module = importlib.import_module(module_name)
            _patch_model_invoke(getattr(module, "structured_llm", None))
        except Exception:
            continue

    return main_module.app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a local MatchaTonic AI server in replay mode without external LLM/RAG calls.",
    )
    parser.add_argument("--repo-root", required=True, help="Target repo root to serve.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "")
    os.environ.setdefault("PINECONE_API_KEY", "")
    os.environ.setdefault("REQUIRE_OPENAI_API_KEY", "false")
    os.environ.setdefault("REQUIRE_PINECONE_FOR_RAG", "false")
    os.environ.setdefault("SPRING_SUMMARY_SYNC_ENABLED", "false")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

    repo_root = Path(args.repo_root).resolve()
    app = _patch_runtime(repo_root)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
