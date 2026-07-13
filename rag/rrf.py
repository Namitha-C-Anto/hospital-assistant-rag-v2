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

    # -------------------------------------------------
    # Compute RRF scores for documents across all
    # ranked retrieval results.
    # -------------------------------------------------
    for ranked_documents in ranked_lists:

        for rank, document in enumerate(ranked_documents):

            # Create a unique identifier for each document.
            key = (
                document.metadata.get("source"),
                document.metadata.get("page"),
                document.page_content,
            )

            # Update the Reciprocal Rank Fusion score.
            scores[key] += 1 / (k + rank + 1)

            # Keep a reference to the original document.
            doc_map[key] = document

    # -------------------------------------------------
    # Sort documents by descending RRF score.
    # -------------------------------------------------
    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    # -------------------------------------------------
    # Return the top-ranked fused documents.
    # -------------------------------------------------
    return [
        doc_map[key]
        for key, _ in ranked[:top_n]
    ]