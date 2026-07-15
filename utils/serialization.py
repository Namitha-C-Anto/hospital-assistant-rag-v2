from langchain_core.documents import Document

def serialize_documents(
    documents: list[Document],
) -> list[dict]:
    
    """
    Convert LangChain documents into a serializable format.

    Args:
        documents: Retrieved or reranked LangChain documents.

    Returns:
        A list of dictionaries containing document rank,
        source, page number, and content.
    """

    return [
    {
        "rank": i + 1,
        "source": document.metadata.get("source"),
        "page": document.metadata.get("page"),
        "content": document.page_content,
    }
    for i, document in enumerate(documents)
]
    