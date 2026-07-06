from collections import defaultdict
from langchain_core.documents import Document

def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
    top_n: int = 15,
) -> list[Document]:
    
    """
    Fuse multiple ranked document lists using Reciprocal Rank Fusion (RRF).

    Args:
        ranked_lists: Ranked lists of retrieved documents.
        k: RRF ranking constant.
        top_n: Number of documents to return.

    Returns:
        A fused ranked list of unique documents.
    """

    scores = defaultdict(float)
    doc_map = {}

    for ranked_docs in ranked_lists:

        for rank, doc in enumerate(ranked_docs):

            # unique key
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content,
            )

            scores[key] += 1 / (k + rank + 1)

            doc_map[key] = doc

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        doc_map[key]
        for key, _ in ranked[:top_n]
    ]