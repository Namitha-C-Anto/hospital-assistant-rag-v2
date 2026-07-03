from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists,
    k=60,
    top_n=15,
):
    """
    ranked_lists: List[List[Document]]
    Returns fused list of Documents.
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