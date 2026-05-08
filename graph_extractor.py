from __future__ import annotations
 
"""
graph_extractor.py
==================
Two-phase LLM pipeline that turns CV chunks into a clean, consistent
knowledge graph schema (entities + relationships).
 
Phase 1 — Per-CV extraction
    For each CV's chunks we call the LLM once and ask it to:
      • identify every meaningful entity (person, skill, company, …)
      • identify every meaningful relationship between those entities
      • invent node labels and relationship types freely — no fixed schema
      • return everything as structured JSON
 
Phase 2 — Cross-document consistency pass
    After every CV has been extracted, one final LLM call sees:
      • the global ontology accumulated so far (all unique labels / rel-types)
      • the raw extraction for the current document
    It returns a "harmonised" version that:
      • reuses existing labels where they mean the same thing
        (e.g. "Degree" vs "Education" → pick one and stick to it)
      • adds genuinely new labels only when nothing existing fits
      • finds cross-document relationships
        (e.g. two candidates who worked at the same company)
 
The final output of this module is a list of GraphDocument objects —
plain Python dataclasses — ready to be written to Neo4j by neo4j_indexing.py.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from llm import get_llm
from prompts import (
    _GRAPH_EXTRACT_PROMPT,
    _GRAPH_CONSISTENCY_PROMPT
)

# Data classes

@dataclass
class GraphEntity:
    """A single node to be created in Neo4j."""
    uid: str                        # stable unique id, e.g. "skill_python"
    label: str                      # Neo4j node label, e.g. "Skill"
    name: str                       # primary display name, e.g. "Python"
    properties: Dict[str, Any] = field(default_factory=dict)
    source_candidate: str = ""      # which CV produced this entity



@dataclass
class GraphRelationship:
    """A single directed edge to be created in Neo4j."""
    from_uid: str                   # uid of the source node
    to_uid: str                     # uid of the target node
    rel_type: str                   # e.g. "HAS_SKILL", "WORKED_AT"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDocument:
    """All entities and relationships extracted from one or more CVs."""
    entities: List[GraphEntity] = field(default_factory=list)
    relationships: List[GraphRelationship] = field(default_factory=list)


# Internal helpers

def _clean_json(raw: str) -> str:
    """Strip markdown fences that LLMs sometimes wrap around JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        # remove opening fence (```json or ```)
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        # remove closing fence
        raw = re.sub(r"```\s*$", "", raw)
    return raw.strip()


def _parse_extraction(raw: str, candidate_name: str) -> Tuple[List[GraphEntity], List[GraphRelationship]]:
    """
    Parse the JSON returned by the extraction / consistency prompts.
 
    Expected shape:
    {
      "entities": [
        {
          "uid": "skill_python",
          "label": "Skill",
          "name": "Python",
          "properties": { "category": "Programming Language" }
        }, ...
      ],
      "relationships": [
        {
          "from_uid": "candidate_ahmed",
          "to_uid": "skill_python",
          "type": "HAS_SKILL",
          "properties": { "proficiency": "advanced" }
        }, ...
      ]
    }
    """
    try:
        data = json.loads(_clean_json(raw))
    except json.JSONDecodeError as exc:
        print(f"[graph_extractor] JSON parse error for {candidate_name}: {exc}")
        print(f"[graph_extractor] Raw output snippet: {raw[:300]}")
        return [], []
 
    entities: List[GraphEntity] = []
    for e in data.get("entities", []):
        uid   = str(e.get("uid", "")).strip()
        label = str(e.get("label", "Entity")).strip()
        name  = str(e.get("name", uid)).strip()
        props = e.get("properties", {})
        if not uid:
            continue
        entities.append(GraphEntity(
            uid=uid,
            label=label,
            name=name,
            properties=props if isinstance(props, dict) else {},
            source_candidate=candidate_name,
        ))

    relationships: List[GraphRelationship] = []
    for r in data.get("relationships", []):
        from_uid = str(r.get("from_uid", "")).strip()
        to_uid   = str(r.get("to_uid", "")).strip()
        rel_type = str(r.get("type", "RELATED_TO")).strip().upper().replace(" ", "_")
        props    = r.get("properties", {})
        if not from_uid or not to_uid:
            continue
        relationships.append(GraphRelationship(
            from_uid=from_uid,
            to_uid=to_uid,
            rel_type=rel_type,
            properties=props if isinstance(props, dict) else {},
        ))

    return entities, relationships


def _build_ontology_summary(graph_doc: GraphDocument) -> str:
    """
    Build a compact text summary of all labels and relationship types
    seen so far.  Passed to the consistency prompt so the LLM knows
    what vocabulary already exists.
    """
    labels    = sorted({e.label for e in graph_doc.entities})
    rel_types = sorted({r.rel_type for r in graph_doc.relationships})

    # Also include a few representative names per label so the LLM
    # understands what each label actually covers.
    label_examples: Dict[str, List[str]] = {}
    for e in graph_doc.entities:
        label_examples.setdefault(e.label, [])
        if e.name not in label_examples[e.label]:
            label_examples[e.label].append(e.name)

    
    lines = ["=== EXISTING ONTOLOGY ===", ""]
    lines.append("Node labels (with example names):")
    for lbl in labels:
        examples = label_examples.get(lbl, [])[:4]
        lines.append(f"  • {lbl}: {', '.join(examples)}")
 
    lines.append("")
    lines.append("Relationship types:")
    for rt in rel_types:
        lines.append(f"  • {rt}")
 
    print(f"{"\n".join(lines)}")
    return "\n".join(lines)


# Public API

def extract_graph_from_chunks(
    chunks: List[Document],
    candidate_name: str,
) -> Tuple[List[GraphEntity], List[GraphRelationship]]:
    """
    Phase 1 — extract entities and relationships from one CV's chunks.
 
    chunks         : all Document objects belonging to one candidate
    candidate_name : used for logging and as the Candidate node's name
 
    Returns (entities, relationships) for this CV only.
    """
    # Assemble the CV text from its chunks, labelled by section
    cv_text_parts = []
    for chunk in chunks:
        section = chunk.metadata.get("section", "General")
        cv_text_parts.append(f"[{section}]\n{chunk.page_content}")
    cv_text = "\n\n".join(cv_text_parts)
 
    llm = get_llm()
    chain = _GRAPH_EXTRACT_PROMPT | llm | StrOutputParser()
 
    print(f"[graph_extractor] Phase 1 — extracting from: {candidate_name}")
    raw = chain.invoke({
        "candidate_name": candidate_name,
        "cv_text": cv_text,
    })
    print(f"raw text\n{raw[:500]}")
    entities, relationships = _parse_extraction(raw, candidate_name)
    print(
        f"[graph_extractor]   → {len(entities)} entities, "
        f"{len(relationships)} relationships"
    )
    return entities, relationships


def harmonise_with_existing(
    new_entities: List[GraphEntity],
    new_relationships: List[GraphRelationship],
    candidate_name: str,
    accumulated: GraphDocument,
) -> Tuple[List[GraphEntity], List[GraphRelationship]]:
    """
    Phase 2 — consistency pass.
 
    Sends the current global ontology + this CV's raw extraction to the LLM
    and asks it to:
      • merge / rename labels and rel-types to match the existing vocabulary
      • discover cross-document relationships if any exist
      • return a harmonised JSON with the same shape as Phase 1
 
    accumulated : the GraphDocument built from all previously processed CVs.
                  Empty on the first CV → LLM just returns the input as-is
                  (there's nothing to be consistent with yet).
    """

    # On the very first CV there's no existing ontology — skip the LLM call
    if not accumulated.entities:
        print(f"[graph_extractor] Phase 2 — first CV, no harmonisation needed.")
        return new_entities, new_relationships
    
    ontology_summary = _build_ontology_summary(accumulated)


    # Serialize the new extraction back to JSON so the LLM can work with it
    new_data = {
        "entities": [
            {
                "uid":        e.uid,
                "label":      e.label,
                "name":       e.name,
                "properties": e.properties,
            }
            for e in new_entities
        ],
        "relationships": [
            {
                "from_uid":   r.from_uid,
                "to_uid":     r.to_uid,
                "type":       r.rel_type,
                "properties": r.properties,
            }
            for r in new_relationships
        ],
    }

    # Also pass a compact list of all *existing* entity names so the LLM
    # can find cross-document matches (same company, same skill name, etc.)
    existing_entities_summary = json.dumps(
        [
            {"uid": e.uid, "label": e.label, "name": e.name}
            for e in accumulated.entities
        ],
        ensure_ascii=False,
        indent=None,
    )
 
    llm = get_llm()
    chain = _GRAPH_CONSISTENCY_PROMPT | llm | StrOutputParser()
 
    print(f"[graph_extractor] Phase 2 — harmonising: {candidate_name}")
    raw = chain.invoke({
        "ontology_summary":        ontology_summary,
        "existing_entities":       existing_entities_summary,
        "candidate_name":          candidate_name,
        "new_extraction":          json.dumps(new_data, ensure_ascii=False, indent=2),
    })
 
    harmonised_entities, harmonised_rels = _parse_extraction(raw, candidate_name)

    print(
        f"[graph_extractor]   → harmonised: {len(harmonised_entities)} entities, "
        f"{len(harmonised_rels)} relationships "
        f"(including possible cross-doc)"
    )
    return harmonised_entities, harmonised_rels


def build_graph_document(
    chunks_per_cv: List[Tuple[str, List[Document]]],
) -> GraphDocument:
    """
    Full two-phase pipeline over all CVs.
 
    chunks_per_cv : list of (candidate_name, [Document, ...]) pairs,
                    one per uploaded CV.
 
    Returns a single GraphDocument containing the complete, harmonised
    set of entities and relationships across all CVs.
    """
    accumulated = GraphDocument()

    # Track UIDs we've already added to avoid duplicates.
    # Harmonisation may return existing-entity UIDs as anchor points for
    # cross-document relationships — we must not re-insert them.
    seen_uids: set[str] = set()
 
    for candidate_name, chunks in chunks_per_cv:
 
        # ── Phase 1: extract from this CV ────────────────────────────────────
        raw_entities, raw_rels = extract_graph_from_chunks(chunks, candidate_name)
 
        # ── Phase 2: harmonise against everything accumulated so far ─────────
        h_entities, h_rels = harmonise_with_existing(
            raw_entities, raw_rels, candidate_name, accumulated
        )
 
        # ── Merge into accumulated graph ──────────────────────────────────────
        for entity in h_entities:
            if entity.uid not in seen_uids:
                accumulated.entities.append(entity)
                seen_uids.add(entity.uid)
            else:
                # Entity already exists (cross-doc dedup) — just update
                # properties if the new extraction added something new.
                existing = next(e for e in accumulated.entities if e.uid == entity.uid)
                for k, v in entity.properties.items():
                    if k not in existing.properties:
                        existing.properties[k] = v
 
        # Relationships are always added (they're directional edges, not nodes)
        # but we deduplicate on (from, type, to) to avoid duplicates
        existing_edges = {
            (r.from_uid, r.rel_type, r.to_uid)
            for r in accumulated.relationships
        }
        for rel in h_rels:
            key = (rel.from_uid, rel.rel_type, rel.to_uid)
            if key not in existing_edges:
                accumulated.relationships.append(rel)
                existing_edges.add(key)
 
    print(
        f"[graph_extractor] Final graph: "
        f"{len(accumulated.entities)} entities, "
        f"{len(accumulated.relationships)} relationships"
    )
    return accumulated    
