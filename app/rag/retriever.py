from typing import Any, Iterable, List

from app.core.config import settings
from app.rag.vectorstore import get_vectorstore


class EmptyRetriever:
    def invoke(self, _: str) -> List[Any]:
        return []


class PineconeFallbackRetriever:
    def __init__(self, vectorstore: Any, current_phase: str, k: int):
        self.vectorstore = vectorstore
        self.current_phase = (current_phase or "").strip()
        self.k = max(1, int(k))

    def _search(self, query: str, with_phase_filter: bool) -> List[Any]:
        search_kwargs: dict[str, Any] = {"k": self.k}
        if with_phase_filter and settings.RAG_PHASE_FILTER_ENABLED and self.current_phase:
            search_kwargs["filter"] = {"phase": self.current_phase}

        try:
            return self.vectorstore.similarity_search(query, **search_kwargs)
        except Exception:
            try:
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs=search_kwargs,
                )
                return retriever.invoke(query)
            except Exception:
                return []

    def invoke(self, query: str) -> List[Any]:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []

        docs = self._search(normalized_query, with_phase_filter=True)
        if docs or not settings.RAG_PHASE_FILTER_ENABLED:
            return docs

        # metadata.phase 가 없거나 phase 값이 다를 때는 필터 없이 재시도
        return self._search(normalized_query, with_phase_filter=False)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _source_of(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(
        metadata.get("source")
        or metadata.get("file")
        or metadata.get("path")
        or metadata.get("url")
        or "unknown"
    )


def format_docs_for_prompt(docs: Iterable[Any]) -> str:
    if not docs:
        return "(관련 레퍼런스를 찾지 못했습니다.)"

    blocks: List[str] = []
    total_chars = 0
    for doc in docs:
        content = " ".join(str(getattr(doc, "page_content", "")).split())
        if not content:
            continue

        source = _source_of(doc)
        snippet = _trim(content, settings.RAG_MAX_DOC_CHARS)
        block = f"- source: {source}\n  snippet: {snippet}"

        if total_chars + len(block) > settings.RAG_MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total_chars += len(block)
        if len(blocks) >= settings.RAG_TOP_K:
            break

    return "\n".join(blocks) if blocks else "(관련 레퍼런스를 찾지 못했습니다.)"


def get_retriever(current_phase: str, k: int | None = None):
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return EmptyRetriever()

    try:
        return PineconeFallbackRetriever(
            vectorstore=vectorstore,
            current_phase=current_phase,
            k=k or settings.RAG_TOP_K,
        )
    except Exception:
        return EmptyRetriever()


def get_rag_context(*, query: str, current_phase: str, k: int | None = None) -> str:
    retriever = get_retriever(current_phase=current_phase, k=k)
    try:
        docs = retriever.invoke(query)
    except Exception:
        docs = []
    return format_docs_for_prompt(docs)
