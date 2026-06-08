from __future__ import annotations

from typing import Any


REQUIRED_EXAMPLE_KEYS = {"inputs", "outputs", "metadata"}
REQUIRED_INPUT_KEYS = {"messages", "response_mode", "backend_schema_name"}
REQUIRED_OUTPUT_KEYS = {
    "expected_behavior",
    "must_include",
    "must_not_include",
    "success_criteria",
}
REQUIRED_METADATA_KEYS = {
    "category",
    "difficulty",
    "split",
    "requires_json",
    "locale",
    "failure_mode",
}

PROJECT_PROGRESS_FIELDS = {
    "project_topic",
    "target_customer",
    "current_decisions",
    "constraints",
    "open_questions",
    "next_steps",
    "member_tasks",
    "risks",
    "source_grounding",
}

CHAT_RESPONSE_FIELDS = {
    "content",
    "suggestedQuestions",
    "currentStatus",
    "isSufficient",
    "collectedData",
}


def validate_dataset_example(example: dict[str, Any], *, line_no: int | None = None) -> None:
    prefix = f"line {line_no}: " if line_no is not None else ""
    missing = REQUIRED_EXAMPLE_KEYS - set(example)
    if missing:
        raise ValueError(f"{prefix}missing top-level keys: {sorted(missing)}")

    inputs = example["inputs"]
    outputs = example["outputs"]
    metadata = example["metadata"]
    if not isinstance(inputs, dict) or not isinstance(outputs, dict) or not isinstance(metadata, dict):
        raise ValueError(f"{prefix}inputs, outputs, and metadata must be objects")

    missing_inputs = REQUIRED_INPUT_KEYS - set(inputs)
    missing_outputs = REQUIRED_OUTPUT_KEYS - set(outputs)
    missing_metadata = REQUIRED_METADATA_KEYS - set(metadata)
    if missing_inputs:
        raise ValueError(f"{prefix}missing input keys: {sorted(missing_inputs)}")
    if missing_outputs:
        raise ValueError(f"{prefix}missing output keys: {sorted(missing_outputs)}")
    if missing_metadata:
        raise ValueError(f"{prefix}missing metadata keys: {sorted(missing_metadata)}")

    messages = inputs["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{prefix}inputs.messages must be a non-empty list")
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{prefix}message {idx} must be an object")
        if message.get("role") not in {"user", "assistant", "system"}:
            raise ValueError(f"{prefix}message {idx} has invalid role")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{prefix}message {idx} content must be a non-empty string")

    for key in ("must_include", "must_not_include", "success_criteria"):
        if not isinstance(outputs[key], list) or not all(isinstance(item, str) for item in outputs[key]):
            raise ValueError(f"{prefix}outputs.{key} must be a list of strings")

    if not isinstance(metadata["split"], list) or not all(isinstance(item, str) for item in metadata["split"]):
        raise ValueError(f"{prefix}metadata.split must be a list of strings")


def validate_project_progress_json(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not an object"
    missing = PROJECT_PROGRESS_FIELDS - set(payload)
    if missing:
        return False, f"missing fields: {sorted(missing)}"
    if not isinstance(payload.get("source_grounding"), dict):
        return False, "source_grounding must be an object"
    return True, "ok"


def validate_chat_response_json(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not an object"
    missing = CHAT_RESPONSE_FIELDS - set(payload)
    if missing:
        return False, f"missing chat response fields: {sorted(missing)}"
    if not isinstance(payload.get("content"), str):
        return False, "content must be a string"
    if not isinstance(payload.get("suggestedQuestions"), list):
        return False, "suggestedQuestions must be a list"
    if not all(isinstance(item, str) for item in payload.get("suggestedQuestions", [])):
        return False, "suggestedQuestions must contain only strings"
    if not isinstance(payload.get("currentStatus"), str):
        return False, "currentStatus must be a string"
    if not isinstance(payload.get("isSufficient"), bool):
        return False, "isSufficient must be a boolean"
    if not isinstance(payload.get("collectedData"), dict):
        return False, "collectedData must be an object"
    return True, "ok"
