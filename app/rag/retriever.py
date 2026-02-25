from typing import Any, List

from app.rag.vectorstore import get_vectorstore


class EmptyRetriever:
    def invoke(self, _: str) -> List[Any]:
        return []


def get_retriever(current_phase: str):
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return EmptyRetriever()

    search_kwargs = {"k": 3, "filter": {"phase": current_phase}}
    try:
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )
    except Exception:
        return EmptyRetriever()
