from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI

from eval.langsmith.schemas import validate_chat_response_json, validate_project_progress_json


def _output_text(outputs: dict[str, Any]) -> str:
    return str(outputs.get("content") or "")


def _extract_json_object(text: str) -> Any | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _output_payload(outputs: dict[str, Any]) -> Any | None:
    if isinstance(outputs, dict):
        return outputs
    return _extract_json_object(str(outputs))


def _schema_candidate(outputs: dict[str, Any]) -> tuple[str, Any | None]:
    payload = _output_payload(outputs)
    if isinstance(payload, dict) and {
        "content",
        "suggestedQuestions",
        "currentStatus",
        "isSufficient",
        "collectedData",
    }.issubset(payload):
        return "chat_response", payload

    content_json = _extract_json_object(_output_text(outputs))
    if content_json is not None:
        return "project_progress_v1", content_json
    return "unknown", payload


def json_parse_pass(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_outputs = reference_outputs or {}
    if not reference_outputs.get("success_criteria"):
        return {"key": "json_parse_pass", "score": 1, "comment": "no JSON criterion in reference"}

    parsed = _output_payload(outputs)
    return {
        "key": "json_parse_pass",
        "score": 1 if parsed is not None else 0,
        "comment": "output is JSON-compatible" if parsed is not None else "no parseable JSON object found",
    }


def json_schema_pass(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema_name, payload = _schema_candidate(outputs)
    if schema_name == "chat_response":
        ok, reason = validate_chat_response_json(payload)
    else:
        ok, reason = validate_project_progress_json(payload)
    return {
        "key": "json_schema_pass",
        "score": 1 if ok else 0,
        "comment": f"{schema_name}: {reason}",
    }


def question_control_pass(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = _output_text(outputs)
    question_count = text.count("?") + text.count("\uff1f")
    return {
        "key": "question_control_pass",
        "score": 1 if question_count <= 2 else 0,
        "comment": f"question_count={question_count}",
    }


def length_control_pass(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text_len = len(_output_text(outputs))
    return {
        "key": "length_control_pass",
        "score": 1 if text_len <= 900 else 0,
        "comment": f"chars={text_len}",
    }


def llm_judge(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    model = os.getenv("LANGSMITH_JUDGE_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = f"""
You are a strict Korean AI product evaluation judge.

Evaluate the assistant output against the reference criteria.
The target under test is the MatchaTonic chat API response, not necessarily a raw
project_progress_v1 object. Judge the useful assistant message in outputs.content,
the extracted structured state in outputs.collectedData, the phase in
outputs.currentStatus, and outputs.suggestedQuestions.

Do not give a score of 1 only because the response is a chat API object instead of
raw project_progress_v1 JSON. Schema validity is checked by separate evaluators.
Focus this score on project guidance, context coverage, conflict resolution,
question control, and hallucination avoidance.

Return JSON only:
{{
  "score": 1,
  "comment": "short Korean explanation",
  "subscores": {{
    "project_guidance": 1,
    "context_coverage": 1,
    "conflict_resolution": 1,
    "hallucination_guard": 1
  }}
}}

Scoring:
5 = excellent, 4 = good, 3 = usable with issues, 2 = poor, 1 = fail.

Input:
{json.dumps(inputs, ensure_ascii=False)}

Reference:
{json.dumps(reference_outputs, ensure_ascii=False)}

Assistant output:
{json.dumps(outputs, ensure_ascii=False)}
"""
    response = llm.invoke(prompt)
    parsed = _extract_json_object(str(response.content))
    if not isinstance(parsed, dict):
        return {"key": "llm_judge", "score": 0, "comment": "judge returned invalid JSON"}
    return {
        "key": "llm_judge",
        "score": float(parsed.get("score", 0)),
        "comment": str(parsed.get("comment", "")),
    }
