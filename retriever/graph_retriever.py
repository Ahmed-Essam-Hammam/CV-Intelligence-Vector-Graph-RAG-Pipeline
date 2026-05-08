from __future__ import annotations
 
"""
graph_retriever.py
==================
Query-time graph retrieval leg.
 
Given a user question:
  1. We call the LLM with the live Neo4j schema + the question and ask it
     to write a READ-ONLY Cypher query.
  2. We execute that query against Neo4j.
  3. We format the result rows into a human-readable string of facts that
     the answer LLM can consume alongside the vector chunks.
 
Safety
------
• Only MATCH / RETURN / WITH / WHERE / OPTIONAL MATCH / CALL (read procedures)
  are allowed.  Any write keyword triggers an immediate refusal.
• The LLM is given the live schema (from neo4j_indexing.get_graph_schema)
  so it knows exactly what labels, properties and relationship types exist.
• If the generated Cypher fails to execute (syntax error, missing label, …)
  we catch the exception and return an empty string — the answer LLM then
  works on vector context only.
"""

import re
from typing import Optional

from neo4j import Driver

from llm import get_llm
from langchain_core.output_parsers import StrOutputParser
from prompts import _GRAPH_CYPHER_PROMPT
from indexing.neo4j_indexing import get_graph_schema
from config import NEO4J_DATABASE, GRAPH_MAX_RESULTS



# Safety guard
_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|CALL\s+apoc\.schema|"
    r"CALL\s+apoc\.create|CALL\s+apoc\.refactor|CALL\s+db\.create)\b",
    re.IGNORECASE,
)
 
 
def _is_safe_cypher(cypher: str) -> bool:
    """Return True if the Cypher contains no write operations."""
    return not bool(_WRITE_KEYWORDS.search(cypher))



# Cypher extraction from LLM output
def _extract_cypher(raw: str) -> str:
    """
    The LLM may wrap the query in markdown fences.
    Extract just the Cypher text.
    """
    raw = raw.strip()

    # Try to extract from ```cypher ... ``` or ``` ... ``` fences
    fence_match = re.search(r"```(?:cypher)?\s*\n?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
 
    # No fences — return as-is (LLM sometimes responds cleanly)
    return raw



# Main retriever
def retrieve_from_graph(
    driver: Driver,
    query: str,
    schema: Optional[str] = None,
    max_results: int = GRAPH_MAX_RESULTS
) -> str:
    """
    Full graph retrieval pipeline.
 
    driver     : active Neo4j driver
    query      : the user's natural-language question
    schema     : pre-computed schema string (pass it in to avoid re-computing
                 on every question); if None, we compute it fresh.
    max_results: cap on the number of rows returned
 
    Returns a formatted string of graph facts, or an empty string if
    retrieval fails or the query returns nothing.
    """

    # Step 1: get schema
    if schema is None:
        schema = get_graph_schema(driver)

    
    # Step 2: ask LLM to generate Cypher
    llm = get_llm()
    chain = _GRAPH_CYPHER_PROMPT | llm | StrOutputParser()

    raw_output = chain.invoke({
        "schema":      schema,
        "question":    query,
        "max_results": max_results,
    })

    cypher = _extract_cypher(raw_output).strip()       

    if not cypher:
        print("[graph_retriever] LLM returned empty Cypher.")
        return ""


    # Step 3: safety check
    if not _is_safe_cypher(cypher):
        print(f"[graph_retriever] Unsafe Cypher blocked:\n{cypher}")
        return ""
 
    print(f"[graph_retriever] Executing Cypher:\n{cypher}")    


    # Step 4: execute against Neo4j
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher)
            records = [dict(record) for record in result]
    except Exception as exc:
        print(f"[graph_retriever] Cypher execution error: {exc}")
        return ""
 
    if not records:
        print("[graph_retriever] Query returned no results.")
        return ""


    # Step 5: format results as readable facts
    facts = _format_records(records)
    print(f"[graph_retriever] Returned {len(records)} graph records.")
    return facts



def _format_records(records: list[dict]) -> str:
    """
    Convert Neo4j result records to a compact, human-readable fact block.
 
    Each record is a dict of {column_name: value}.  Values can be:
      • scalar (str, int, float, bool)
      • Neo4j Node  → we render its properties dict
      • Neo4j Relationship → we render its type + properties
      • list of the above
 
    Output format:
        === GRAPH FACTS ===
        [1] candidate_name: Ahmed Essam | skill: Python | proficiency: Advanced
        [2] candidate_name: Malak Soula | skill: Machine Learning | ...
        ...
    """
    lines = ["=== GRAPH FACTS ==="]
 
    for i, record in enumerate(records, 1):
        parts = []
        for key, value in record.items():
            rendered = _render_value(value)
            parts.append(f"{key}: {rendered}")
        lines.append(f"[{i}] " + " | ".join(parts))
 
    return "\n".join(lines)



def _render_value(value) -> str:
    """Recursively render a Neo4j result value to a string."""
    if value is None:
        return "N/A"
 
    # Neo4j Node object — has .items() like a dict of properties
    if hasattr(value, "items") and hasattr(value, "labels"):
        labels = ":".join(value.labels)
        props  = {k: v for k, v in value.items() if k != "uid"}
        prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
        return f"({labels} {prop_str})" if prop_str else f"({labels})"
 
    # Neo4j Relationship object
    if hasattr(value, "type") and hasattr(value, "start_node"):
        props    = {k: v for k, v in value.items()}
        prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
        return f"[:{value.type} {prop_str}]" if prop_str else f"[:{value.type}]"
 
    # List
    if isinstance(value, list):
        return "[" + ", ".join(_render_value(v) for v in value) + "]"
 
    return str(value)