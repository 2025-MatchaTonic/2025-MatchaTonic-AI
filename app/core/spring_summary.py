import json
import logging
from typing import Mapping
from urllib import error, request

from app.ai.graph.collected_data import sanitize_collected_data
from app.core.config import settings

logger = logging.getLogger(__name__)


class SpringSummarySyncError(RuntimeError):
    pass


def _clean_optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def build_spring_summary_payload(collected_data: Mapping[str, object] | None) -> dict[str, str]:
    sanitized = sanitize_collected_data(collected_data)
    payload: dict[str, str] = {}

    subject = _clean_optional_string(sanitized.get("subject"))
    title = _clean_optional_string(sanitized.get("title"))
    if subject:
        payload["subject"] = subject
    if title:
        payload["name"] = title
        payload["title"] = title
    elif subject:
        payload["name"] = subject

    goal = _clean_optional_string(sanitized.get("goal"))
    if goal:
        payload["goal"] = goal

    team_size = sanitized.get("teamSize")
    if team_size is not None:
        payload["teamSize"] = str(team_size)

    roles = sanitized.get("roles")
    if isinstance(roles, list) and roles:
        payload["roles"] = ", ".join(str(role).strip() for role in roles if str(role).strip())

    due_date = _clean_optional_string(sanitized.get("dueDate"))
    if due_date:
        payload["dueDate"] = due_date

    deliverables = _clean_optional_string(sanitized.get("deliverables"))
    if deliverables:
        payload["deliverables"] = deliverables

    return payload


def build_spring_summary_headers(authorization: str | None = None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    cleaned_authorization = _clean_optional_string(authorization)
    if cleaned_authorization:
        headers["Authorization"] = cleaned_authorization
        return headers

    configured_token = _clean_optional_string(settings.SPRING_AUTH_BEARER_TOKEN)
    if configured_token:
        headers["Authorization"] = f"Bearer {configured_token}"

    return headers


def build_spring_summary_url(project_id: int | str) -> str:
    base_url = _clean_optional_string(settings.SPRING_API_BASE_URL)
    if not base_url:
        raise SpringSummarySyncError("SPRING_API_BASE_URL is not configured.")

    path = settings.SPRING_SUMMARY_PATH_TEMPLATE.format(project_id=project_id)
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def sync_project_summary(
    project_id: int | str,
    collected_data: Mapping[str, object] | None,
    *,
    authorization: str | None = None,
) -> None:
    if not settings.SPRING_SUMMARY_SYNC_ENABLED:
        return

    payload = build_spring_summary_payload(collected_data)
    if not payload:
        logger.info("spring summary sync skipped project_id=%s reason=empty_payload", project_id)
        return

    url = build_spring_summary_url(project_id)
    headers = build_spring_summary_headers(authorization)
    encoded_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    logger.info(
        "spring summary sync request project_id=%s url=%s payload=%s",
        project_id,
        url,
        json.dumps(payload, ensure_ascii=False),
    )

    req = request.Request(
        url=url,
        data=encoded_body,
        headers=headers,
        method="PATCH",
    )

    try:
        with request.urlopen(req, timeout=settings.SPRING_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            logger.info(
                "spring summary sync response project_id=%s status=%s body=%s",
                project_id,
                getattr(response, "status", "unknown"),
                response_body[:500],
            )
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        logger.exception(
            "spring summary sync failed project_id=%s status=%s body=%s",
            project_id,
            exc.code,
            response_body[:500],
        )
        raise SpringSummarySyncError(
            f"Spring summary sync failed with status {exc.code}."
        ) from exc
    except error.URLError as exc:
        logger.exception("spring summary sync network error project_id=%s", project_id)
        raise SpringSummarySyncError("Spring summary sync network error.") from exc
