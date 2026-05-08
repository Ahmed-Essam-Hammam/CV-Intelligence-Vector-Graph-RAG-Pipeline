from __future__ import annotations

from typing import List, Optional

from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny
)

from config import EMBEDDING_MODEL, EMBEDDING_DIM, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from embeddings import get_embeddings



def _embed(texts: List[str]) -> List[List[float]]:
    from embeddings import get_embeddings
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)


_qdrant_client = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
    return _qdrant_client


def build_qdrant_index() -> QdrantClient:
    """
    Create (or recreate) the Qdrant collection with cosine similarity.
    Wipes existing data so each app session starts clean.
    """

    client = get_qdrant_client()

    # Delete if exists — ensures a clean slate on every upload
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name = QDRANT_COLLECTION,
        vectors_config = VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )

    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="candidate_name",
        field_schema="keyword",
    )

    return client



def add_documents_to_qdrant(
    client: QdrantClient,
    docs: List[Document],
) -> None:
    """
    Embed all chunks and upsert into Qdrant.

    Each point stores:
      vector   : normalised embedding of page_content
      payload  : candidate_name, section, source_cv, page_content
                 (page_content in payload lets us reconstruct the Document
                  on retrieval without a separate doc store)
    """

    if not docs:
        return
    
    texts = [doc.page_content for doc in docs]
    vectors = _embed(texts)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "page_content":   doc.page_content,
                "candidate_name": doc.metadata.get("candidate_name", "Unknown"),
                "section":        doc.metadata.get("section", ""),
                "source_cv":      doc.metadata.get("source_cv", ""),
            },
        )
        for i, doc in enumerate(docs)
    ]

    # Upsert in batches of 100 to stay within Qdrant Cloud free-tier limits
    batch_size = 20
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[i : i + batch_size]
        )


# ── Dense search

def search_qdrant(
    client: QdrantClient,
    docs: List[Document],
    query: str,
    k: int,
    allowed_indices: Optional[set] = None,
) -> List[Document]:
    
    """
    Cosine-similarity top-k search against Qdrant.

    Returns a list of Document objects reconstructed from Qdrant payloads.
    """

    query_vector = get_embeddings().embed_query(query)

    # Build candidate_name filter from allowed_indices if provided
    qdrant_filter = None
    if allowed_indices is not None:
        candidate_names = list({
            docs[i].metadata.get("candidate_name", "")
            for i in allowed_indices
            if i < len(docs)
        })
        if candidate_names:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="candidate_name",
                        match=MatchAny(any=candidate_names),
                    )
                ]
            )

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=k,
        query_filter=qdrant_filter,
        with_payload=True,
    ).points

    return [
        Document(
            page_content=r.payload["page_content"],
            metadata={
                "candidate_name": r.payload.get("candidate_name", "Unknown"),
                "section":        r.payload.get("section", ""),
                "source_cv":      r.payload.get("source_cv", ""),
            },
        )
        for r in results
    ]

