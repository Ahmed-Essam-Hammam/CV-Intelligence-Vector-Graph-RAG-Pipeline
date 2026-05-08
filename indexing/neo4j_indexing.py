from __future__ import annotations
"""
neo4j_indexing.py
=================
Deterministic Neo4j writer.
 
No LLM is involved here.  We receive clean Python dataclasses from
graph_extractor.py and translate them into Cypher MERGE statements
executed against Neo4j.
 
Design choices
--------------
• MERGE on uid prevents duplicate nodes when re-indexing.
• Node labels are dynamic (set via APOC apoc.create.addLabels or
  using a string-interpolated label — we use the safer pre-validated
  string approach since labels come from our own LLM pipeline).
• Relationship types are also dynamic; Neo4j does not support
  parameterised relationship types, so we build those Cypher strings
  in Python after sanitising the type string.
• All string properties are stored as Neo4j string properties.
• We wipe the database on every indexing run (clean slate per session),
  matching the behaviour of build_qdrant_index().
"""

import re
from typing import List, Optional

from neo4j import GraphDatabase, Driver

from config import NEO4J_URI, NEO4J_USER, NEO4J_DATABASE, NEO4J_PASSWORD
from graph_extractor import GraphDocument, GraphEntity, GraphRelationship



# Neo4j client singleton

_driver: Optional[Driver] = None

def get_neo4j_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        _driver.verify_connectivity()
        print("[neo4j] Connected to Neo4j.")
    return _driver


def close_neo4j_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None




# Safety helpers  

def _sanitise_label(label: str) -> str:
    """
    Neo4j node labels must start with a letter and contain only
    alphanumeric characters or underscores.
    We also title-case them for readability.
    """
    # Remove anything that isn't alphanumeric or space/underscore
    clean = re.sub(r"[^A-Za-z0-9_ ]", "", label).strip()
    # Title-case each word, then join without spaces
    clean = "".join(word.capitalize() for word in clean.split())
    if not clean:
        clean = "Entity"
    # Must start with a letter
    if not clean[0].isalpha():
        clean = "Node" + clean
    return clean


def _sanitise_rel_type(rel_type: str) -> str:
    """
    Neo4j relationship types must be uppercase with underscores.
    """
    clean = re.sub(r"[^A-Za-z0-9_]", "_", rel_type).upper().strip("_")
    if not clean:
        clean = "RELATED_TO"
    return clean


def _sanitise_properties(props: dict) -> dict:
    """
    Keep only JSON-serialisable scalar values.
    Convert everything to str / int / float / bool.
    """
    safe = {}
    for k, v in props.items():
        key = re.sub(r"[^A-Za-z0-9_]", "_", str(k)).strip("_") or "prop"
        if isinstance(v, (str, int, float, bool)):
            safe[key] = v
        elif v is None:
            pass  # skip nulls — Neo4j stores absence rather than null
        else:
            safe[key] = str(v)
    return safe




# Database lifecycle

def wipe_graph(driver: Driver) -> None:
    """
    Delete all nodes and relationships.
    Uses CALL { } IN TRANSACTIONS for large graphs to avoid memory issues.
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(
            "CALL { MATCH (n) DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS"
        )
    print("[neo4j] Graph wiped.")


def create_constraints(driver: Driver, labels: List[str]) -> None:
    """
    Create uniqueness constraints on uid for every node label we'll use.
    This also creates an index, speeding up MERGE lookups.
    Constraints are idempotent in Neo4j 5.x (CREATE CONSTRAINT IF NOT EXISTS).
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        for label in labels:
            safe = _sanitise_label(label)
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS "
                f"FOR (n:{safe}) REQUIRE n.uid IS UNIQUE"
            )
    print(f"[neo4j] Constraints ensured for {len(labels)} labels.")




# Node writer

def _write_nodes_batch(
    driver: Driver,
    entities: List[GraphEntity],
    batch_size: int = 100,
) -> None:
    """
    MERGE each entity as a node.
    We group by label so each batch uses a single label in the Cypher.
    """

    # Group by sanitised label
    by_label: dict[str, List[GraphEntity]] = {}
    for entity in entities:
        lbl = _sanitise_label(entity.label)
        by_label.setdefault(lbl, []).append(entity)

    
    with driver.session(database=NEO4J_DATABASE) as session:
        for lbl, group in by_label.items():
            for i in range(0, len(group), batch_size):
                batch = group[i : i + batch_size]
                rows = []
                for e in batch:
                    props = _sanitise_properties(e.properties)
                    props["uid"]  = e.uid
                    props["name"] = e.name
                    if e.source_candidate:
                        props["source_candidate"] = e.source_candidate
                    rows.append(props)
 
                # Cypher: UNWIND rows, MERGE on uid, SET all properties
                cypher = (
                    f"UNWIND $rows AS row "
                    f"MERGE (n:{lbl} {{uid: row.uid}}) "
                    f"SET n += row"
                )
                session.run(cypher, rows=rows)
 
    print(f"[neo4j] Wrote {len(entities)} nodes across {len(by_label)} labels.")



# Relationship writer

def _write_relationships_batch(
    driver: Driver,
    relationships: List[GraphRelationship],
    batch_size: int = 100,
) -> None:
    """
    MERGE each relationship.
    Because Neo4j doesn't support parameterised relationship types,
    we group by rel_type and issue one Cypher template per type.
    """
    # Group by sanitised rel_type
    by_type: dict[str, List[GraphRelationship]] = {}
    for rel in relationships:
        rt = _sanitise_rel_type(rel.rel_type)
        by_type.setdefault(rt, []).append(rel)
 
    with driver.session(database=NEO4J_DATABASE) as session:
        for rt, group in by_type.items():
            for i in range(0, len(group), batch_size):
                batch = group[i : i + batch_size]
                rows = []
                for r in batch:
                    props = _sanitise_properties(r.properties)
                    props["from_uid"] = r.from_uid
                    props["to_uid"]   = r.to_uid
                    rows.append(props)
 
                # We MATCH both endpoints by uid across ALL node labels
                # (using a label-agnostic match), then MERGE the relationship.
                cypher = (
                    f"UNWIND $rows AS row "
                    f"MATCH (a {{uid: row.from_uid}}) "
                    f"MATCH (b {{uid: row.to_uid}}) "
                    f"MERGE (a)-[r:{rt}]->(b) "
                    f"SET r += row"
                )
                session.run(cypher, rows=rows)
 
    written = sum(len(v) for v in by_type.values())
    print(
        f"[neo4j] Wrote {written} relationships "
        f"across {len(by_type)} types."
    )



# Public API

def build_neo4j_graph(graph_doc: GraphDocument) -> Driver:
    """
    Full indexing pipeline:
      1. Connect (or reuse connection)
      2. Wipe existing data (clean slate)
      3. Create uniqueness constraints for every label
      4. Write all nodes
      5. Write all relationships
 
    Returns the driver so the caller can pass it to graph_retriever.py.
    """
    driver = get_neo4j_driver()
 
    # Clean slate
    wipe_graph(driver)
 
    # Unique labels for constraint creation
    unique_labels = list({e.label for e in graph_doc.entities})
    create_constraints(driver, unique_labels)
 
    # Write nodes first, then edges (edges reference node uids)
    _write_nodes_batch(driver, graph_doc.entities)
    _write_relationships_batch(driver, graph_doc.relationships)
 
    print("[neo4j] Graph build complete.")
    return driver



def get_graph_schema(driver: Driver) -> str:
    """
    Introspect Neo4j to build a compact schema description:
      - all node labels with their property keys
      - all relationship types with source → target label patterns
 
    This is passed to the graph retriever's Cypher generation prompt
    so the LLM knows exactly what's in the database.
    """
    with driver.session(database=NEO4J_DATABASE) as session:
 
        # Node labels + property keys
        label_props: dict[str, set[str]] = {}
        result = session.run(
            "CALL apoc.meta.schema() YIELD value RETURN value"
        )
        # apoc.meta.schema() may not be available on all Neo4j setups.
        # Fall back to a simpler introspection if needed.
        try:
            schema_value = result.single()["value"]
            for label, info in schema_value.items():
                if info.get("type") == "node":
                    label_props[label] = set(info.get("properties", {}).keys())
        except Exception:
            # Fallback: collect labels and properties manually
            label_props = {}
            lbl_result = session.run("CALL db.labels() YIELD label RETURN label")
            for record in lbl_result:
                lbl = record["label"]
                prop_result = session.run(
                    "MATCH (n) WHERE $lbl IN labels(n) "
                    "UNWIND keys(n) AS k RETURN DISTINCT k LIMIT 20",
                    lbl=lbl,
                )
                label_props[lbl] = {r["k"] for r in prop_result}
 
        # Relationship patterns  (source_label)-[TYPE]->(target_label)
        rel_result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN DISTINCT labels(a)[0] AS src, type(r) AS rel, labels(b)[0] AS tgt "
            "LIMIT 200"
        )
        rel_patterns = [
            f"({r['src']})-[:{r['rel']}]->({r['tgt']})"
            for r in rel_result
        ]
 
    lines = ["=== NEO4J GRAPH SCHEMA ===", ""]
    lines.append("Node labels and their properties:")
    for lbl, props in sorted(label_props.items()):
        lines.append(f"  • {lbl}: {', '.join(sorted(props)) or '(no extra props)'}")
 
    lines.append("")
    lines.append("Relationship patterns:")
    for pat in sorted(set(rel_patterns)):
        lines.append(f"  • {pat}")
 
    return "\n".join(lines)