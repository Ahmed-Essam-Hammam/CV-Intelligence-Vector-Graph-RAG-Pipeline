from __future__ import annotations

import re
from typing import List, Optional, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document



def _tokenize(text: str) -> List[str]:
    """
    Lowercase and split on non-alphanumeric characters.
    Simple but effective for CV text.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(docs: List[Document]) -> BM25Okapi:
    """
    Build and return a BM25 index from the list of Document chunks.
    Returns the index so the caller can store and pass it explicitly —
    avoids module-level singleton issues with Streamlit cache_resource.
    """
    tokens = [_tokenize(doc.page_content) for doc in docs]
    return BM25Okapi(tokens)


def search_bm25(
    bm25_index: BM25Okapi,
    docs: List[Document],
    query: str,
    k: int,
    allowed_indices: Optional[set] = None,
) -> List[Tuple[Document, int]]:
    """
    BM25 search returning (Document, global_index) tuples.

    bm25_index     : BM25Okapi instance returned by build_bm25_index().
    docs           : the same list passed to build_bm25_index().
    allowed_indices: restrict results to this set of global positions.
    """
    tokens = _tokenize(query)
    scores = bm25_index.get_scores(tokens)

    candidates = [
        (scores[i], i)
        for i in range(len(docs))
        if allowed_indices is None or i in allowed_indices
    ]
    candidates.sort(reverse=True)

    return [(docs[i], i) for _, i in candidates[:k]]