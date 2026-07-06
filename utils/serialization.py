from langchain_core.documents import Document

def serialize_documents(
    docs: list[Document],) -> list[dict]:
    return [
    {
        "rank": i + 1,
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page"),
        "content": doc.page_content,
    }
    for i, doc in enumerate(docs)
]
    