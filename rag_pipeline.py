from __future__ import annotations

"""
rag_pipeline.py
===============
Orchestrates the full Graph + Hybrid RAG pipeline.

Indexing
--------
create_indexes()
  1. Chunk CVs (LLM or Docling+NER)
  2. Build Qdrant dense index + BM25 sparse index  (vector leg)
  3. Extract graph entities/relationships from chunks  (graph leg)
  4. Write the graph to Neo4j
  5. Pre-compute and cache the Neo4j schema string

Query
-----
answer_query()
  In parallel:
    A. Graph leg  — LLM generates Cypher → Neo4j → structured facts
    B. Vector leg — Hybrid (Dense + Sparse) retrieval → top-k chunks
  Combined context → LLM answer
"""

import json
from typing import List, Tuple, Optional, Dict

from neo4j import Driver
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from indexing.qdrant_indexing import build_qdrant_index, add_documents_to_qdrant, get_qdrant_client, build_qdrant_index
from retriever.bm25_retriever import build_bm25_index
from retriever.hybrid_retriever import retrieve
from llm import get_llm
from chunking import chunk_cvs_with_llm, chunk_cvs_with_docling
from graph_extractor import build_graph_document
from indexing.neo4j_indexing import build_neo4j_graph, get_graph_schema, get_neo4j_driver, wipe_graph
from retriever.graph_retriever import retrieve_from_graph
from prompts import (
    _ANSWER_PROMPT,
    _EXTRACT_NAME_PROMPT,
    _REWRITE_QUERY_PROMPT,
)
from config import (
    CHUNKS_PER_QUERY_LLM,
    CHUNKS_PER_QUERY_DOCLING,
    RETRIEVAL_MODE,
    GRAPH_MAX_RESULTS,
)


# Indexing 

def load_existing_indexes(
    chunking_mode: str = "LLM Chunking",
) -> Tuple[QdrantClient, object, Driver, str, List[Document], int]:
    """
    Connect to existing Qdrant collection and Neo4j graph WITHOUT wiping
    or re-indexing anything.  Called on startup when indexes already exist.

    Reconstructs all_chunks from the Qdrant payloads so BM25 stays in sync.
    Returns the same tuple shape as create_indexes().
    """
    from qdrant_indexing import get_qdrant_client
    from neo4j_indexing import get_neo4j_driver, get_graph_schema
    from config import QDRANT_COLLECTION

    # Qdrant
    qdrant_client = get_qdrant_client()

    # Reconstruct chunks from stored Qdrant payloads
    all_chunks: List[Document] = []
    offset = None
    while True:
        results, offset = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            p = point.payload
            all_chunks.append(Document(
                page_content=p.get("page_content", ""),
                metadata={
                    "candidate_name": p.get("candidate_name", "Unknown"),
                    "section":        p.get("section", ""),
                    "source_cv":      p.get("source_cv", ""),
                },
            ))
        if offset is None:
            break

    bm25_index = build_bm25_index(all_chunks)

    # Neo4j
    neo4j_driver = get_neo4j_driver()
    graph_schema = get_graph_schema(neo4j_driver)

    k = CHUNKS_PER_QUERY_DOCLING if chunking_mode == "Docling + NER" else CHUNKS_PER_QUERY_LLM

    print(f"[rag_pipeline] Loaded existing indexes — {len(all_chunks)} chunks from Qdrant.")
    return qdrant_client, bm25_index, neo4j_driver, graph_schema, all_chunks, k


def clear_all_indexes() -> None:
    """
    Wipe Qdrant collection and Neo4j graph.
    Called explicitly by the UI "Clear & Rebuild" button.
    """

    print("[rag_pipeline] Clearing all indexes...")
    build_qdrant_index()          # deletes + recreates the collection
    wipe_graph(get_neo4j_driver())
    print("[rag_pipeline] All indexes cleared.")


def create_indexes(
    documents_per_cv: List[List[Document]],
    chunking_mode: str = "LLM Chunking",
) -> Tuple[QdrantClient, object, Driver, str, List[Document], int]:
    """
    Full indexing pipeline — vector + graph.

    Parameters
    ----------
    documents_per_cv : one list of Documents per uploaded CV
    chunking_mode    : "LLM Chunking" or "Docling + NER"

    Returns
    -------
    qdrant_client  : for dense vector search
    bm25_index     : for sparse keyword search
    neo4j_driver   : for graph queries
    graph_schema   : pre-computed schema string (passed to Cypher generator)
    all_chunks     : flat list of all chunk Documents (needed for BM25)
    k              : chunks-per-query for this chunking mode
    """

    # Step 1: Chunk
    if chunking_mode == "Docling + NER":
        all_chunks = chunk_cvs_with_docling(documents_per_cv)
        k = CHUNKS_PER_QUERY_DOCLING
    else:
        all_chunks = chunk_cvs_with_llm(documents_per_cv)
        k = CHUNKS_PER_QUERY_LLM

    if not all_chunks:
        raise ValueError(
            "No chunks were produced from the uploaded CVs. "
            "Check that the files are readable and the chunking pipeline ran correctly."
        )

    # Step 2: Vector indexes 
    # build_qdrant_index() wipes + recreates the collection.
    # The UI calls clear_all_indexes() before create_indexes(), so this is
    # always starting into a clean collection.
    qdrant_client = build_qdrant_index()
    add_documents_to_qdrant(qdrant_client, all_chunks)
    bm25_index = build_bm25_index(all_chunks)

    # Step 3: Group chunks by candidate for graph extraction
    chunks_by_candidate: Dict[str, List[Document]] = {}
    for chunk in all_chunks:
        name = chunk.metadata.get("candidate_name", "Unknown")
        chunks_by_candidate.setdefault(name, []).append(chunk)

    # Maintain insertion order (same order as CVs were uploaded)
    ordered_candidates = list(dict.fromkeys(
        chunk.metadata.get("candidate_name", "Unknown")
        for chunk in all_chunks
    ))
    chunks_per_cv = [
        (name, chunks_by_candidate[name])
        for name in ordered_candidates
    ]

    # Step 4: Graph extraction + Neo4j write
    graph_doc = build_graph_document(chunks_per_cv)
    neo4j_driver = build_neo4j_graph(graph_doc)

    # Step 5: Pre-compute schema for query time
    graph_schema = get_graph_schema(neo4j_driver)
    print("[rag_pipeline] Graph schema cached.")
    print(graph_schema)

    return qdrant_client, bm25_index, neo4j_driver, graph_schema, all_chunks, k


# Candidate-aware vector retrieval helpers (kept from original)

def _extract_candidate_name_from_query(
    query: str,
    known_candidates: List[str],
) -> Optional[List[str]]:
    """Ask the LLM whether the query targets one or more specific candidates."""
    llm = get_llm()
    chain = _EXTRACT_NAME_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "query":      query,
        "candidates": ", ".join(known_candidates),
    }).strip()

    if raw.upper() == "NONE" or not raw:
        return None

    raw_names = [n.strip() for n in raw.split(",") if n.strip()]
    matched = []
    for raw_name in raw_names:
        raw_lower = raw_name.lower()
        exact = next((n for n in known_candidates if n.lower() == raw_lower), None)
        if exact:
            matched.append(exact)
            continue
        partial = next(
            (n for n in known_candidates
             if raw_lower in n.lower() or n.lower() in raw_lower),
            None,
        )
        if partial and partial not in matched:
            matched.append(partial)

    return matched or None


def _rewrite_query_without_name(query: str, candidate_name: str) -> str:
    """Strip the candidate name from the query so the embedding is topic-focused."""
    llm = get_llm()
    chain = _REWRITE_QUERY_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({
        "query":          query,
        "candidate_name": candidate_name,
    }).strip()
    return rewritten if rewritten else query


def _filter_docs_by_candidate(
    docs: List[Document],
    candidate_name: str,
) -> Tuple[List[Document], set]:
    filtered = []
    allowed_indices = set()
    for i, doc in enumerate(docs):
        if doc.metadata.get("candidate_name", "").lower() == candidate_name.lower():
            filtered.append(doc)
            allowed_indices.add(i)
    return filtered, allowed_indices


# Vector retrieval

def _vector_retrieve(
    qdrant_client: QdrantClient,
    bm25_index,
    all_chunks: List[Document],
    query: str,
    k: int,
    retrieval_mode: str = RETRIEVAL_MODE,
) -> List[Document]:
    """
    Candidate-aware hybrid retrieval (no multi-query).

    If a candidate name is detected, search is restricted to that
    candidate's chunks and the query is rewritten without the name.
    """
    known_candidates = list({
        doc.metadata.get("candidate_name", "")
        for doc in all_chunks
        if doc.metadata.get("candidate_name")
    })

    target_candidates = _extract_candidate_name_from_query(query, known_candidates)

    if target_candidates:
        allowed_indices: Optional[set] = set()
        for name in target_candidates:
            _, indices = _filter_docs_by_candidate(all_chunks, name)
            allowed_indices |= indices

        search_query = _rewrite_query_without_name(query, target_candidates[0])
        for name in target_candidates[1:]:
            search_query = _rewrite_query_without_name(search_query, name)
    else:
        allowed_indices = None
        search_query = query

    results = retrieve(
        index=qdrant_client,
        bm25_index=bm25_index,
        docs=all_chunks,
        query=search_query,
        k=k,
        retrieval_mode=retrieval_mode,
        allowed_indices=allowed_indices,
    )
    return results


# Main answer entry point

def answer_query(
    qdrant_client: QdrantClient,
    bm25_index,
    neo4j_driver: Driver,
    graph_schema: str,
    all_chunks: List[Document],
    user_query: str,
    k: int,
    retrieval_mode: str = RETRIEVAL_MODE,
) -> Tuple[str, str, List[Document]]:
    """
    Full Graph + Hybrid RAG pipeline.

    Returns
    -------
    answer         : LLM-generated answer string
    graph_context  : formatted graph facts string (for display in UI)
    vector_docs    : retrieved chunk Documents (for display in UI)
    """

    # Leg A: Graph retrieval
    graph_context = retrieve_from_graph(
        driver=neo4j_driver,
        query=user_query,
        schema=graph_schema,
        max_results=GRAPH_MAX_RESULTS,
    )

    # Leg B: Vector retrieval
    vector_docs = _vector_retrieve(
        qdrant_client=qdrant_client,
        bm25_index=bm25_index,
        all_chunks=all_chunks,
        query=user_query,
        k=k,
        retrieval_mode=retrieval_mode,
    )
    vector_context = "\n\n---\n\n".join(doc.page_content for doc in vector_docs)

    # Answer generation
    llm   = get_llm()
    chain = _ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "graph_context":  graph_context or "No graph data retrieved.",
        "vector_context": vector_context or "No vector chunks retrieved.",
        "question":       user_query,
    })

    return answer, graph_context, vector_docs