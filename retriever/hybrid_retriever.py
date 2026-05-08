from __future__ import annotations

from typing import List, Optional
from langchain_core.documents import Document

from indexing.qdrant_indexing import search_qdrant
from retriever.bm25_retriever import search_bm25
from qdrant_client import QdrantClient

from config import RRF_K


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    rrf_k: int = RRF_K,
) -> List[Document]:
    """
    Merge multiple ranked Document lists using Reciprocal Rank Fusion.

    ranked_lists : one ranked list per retriever (or per query).
    rrf_k        : smoothing constant (standard default: 60).

    Returns a single merged list sorted by descending RRF score,
    de-duplicated by page_content.
    """

    scores = {}
    doc_map = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content
            scores[key]  = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_map[key] = doc


    merged_keys = sorted(doc_map.keys(), key=lambda key: scores[key], reverse=True)
    return [doc_map[key] for key in merged_keys]


def retrieve(
    index: QdrantClient,
    bm25_index: BM25Okapi,
    docs: List[Document],
    query: str,
    k: int,
    retrieval_mode: str = "Hybrid",
    allowed_indices=None,
) -> List[Document]:
    """
    Single entry point for all retrieval modes.

    index      : QdrantClient for dense search.
    bm25_index : BM25Okapi instance for sparse search.
    docs       : full chunk list (needed for BM25 and candidate filtering).
    """
    if retrieval_mode == "Dense":
        return search_qdrant(index, docs, query, k, allowed_indices=allowed_indices)

    if retrieval_mode == "Sparse":
        return [doc for doc, _ in search_bm25(bm25_index, docs, query, k, allowed_indices=allowed_indices)]

    # Hybrid — run both, merge via RRF
    dense_results  = search_qdrant(index, docs, query, k, allowed_indices=allowed_indices)
    sparse_results = [doc for doc, _ in search_bm25(bm25_index, docs, query, k, allowed_indices=allowed_indices)]

    merged = reciprocal_rank_fusion([dense_results, sparse_results])
    return merged[:k]
