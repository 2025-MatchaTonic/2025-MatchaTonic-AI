from app.rag.vectorstore import get_vectorstore


def get_retriever(current_phase: str):
    vectorstore = get_vectorstore()

    # 해당 단계(예: "1")의 메타데이터를 가진 문서만 2개 검색
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 2,
            # "filter": {"phase": current_phase} # 코랩 전처리 시 넣은 태그가 있다면 주석 해제!
        },
    )
    return retriever
