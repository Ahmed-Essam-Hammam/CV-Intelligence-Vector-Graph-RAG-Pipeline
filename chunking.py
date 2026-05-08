from __future__ import annotations

import json
import numpy as np
from typing import List, Tuple
import re

from sentence_transformers import util as st_util
from sentence_transformers import SentenceTransformer

from docling.document_converter import DocumentConverter
import pdfplumber

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from llm import get_llm
from prompts import _PARSE_CV_PROMPT
from config import EMBEDDING_MODEL, NER_MODEL, KNOWN_SECTIONS, HEADERS_EMBEDDING_MODEL





def parse_cv_with_llm(cv_text: str) -> Tuple[str, List[dict]]:
    """
    Send the full CV text to the LLM and get back:
      - candidate_name : the extracted full name
      - chunks         : list of {section, content} dicts

    Falls back gracefully if JSON parsing fails.
    """

    llm = get_llm()
    chain = _PARSE_CV_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"cv_text": cv_text}).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        name = parsed.get("candidate_name", "Unknown Candidate").strip()
        chunks = parsed.get("chunks", [])
        return name, chunks
    except json.JSONDecodeError:
        # Fallback: treat the entire CV as one chunk
        return "Unknown Candidate", [{"section": "Full CV", "content": cv_text}]
    


def chunk_cvs_with_llm(
    documents_per_cv: List[List[Document]],
) -> List[Document]:
    """
    For each CV:
      1. Concatenate all its pages into one text.
      2. Ask the LLM to extract the candidate name AND split into sections.
      3. Create one Document per section, with metadata:
           - candidate_name : real name from LLM
           - section        : e.g. "Skills", "Work Experience"
           - source_cv      : original filename
      4. Prefix each chunk text with "Candidate: <name> | Section: <section>"
         so the embedding carries both identity and topic signals.

    Returns a flat list of all chunk Documents across all CVs.
    """

    all_chunks: List[Document] = []

    for cv_docs in documents_per_cv:
        source_cv = cv_docs[0].metadata.get("source_cv", "unknown")
        cv_text = "\n\n".join(doc.page_content for doc in cv_docs)

        candidate_name, sections = parse_cv_with_llm(cv_text)

        for section in sections:
            section_name = section.get("section", "General")
            content = section.get("content", "").strip()
            if not content:
                continue

            prefixed_content = (
                f"Candidate: {candidate_name} | Section: {section_name}\n\n{content}"
            )

            chunk = Document(
                page_content=prefixed_content,
                metadata={
                    "candidate_name": candidate_name,
                    "section": section_name,
                    "source_cv": source_cv,
                },
            )
            all_chunks.append(chunk)

    return all_chunks



_ner_pipeline = None


def get_ner_pipeline():
    """
    Load the RoBERTa-large NER model once and cache it for the session.
    Jean-Baptiste/roberta-large-ner-english recognises PER, ORG, LOC, MISC.
    """

    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline as hf_pipeline
        _ner_pipeline = hf_pipeline(
            "ner",
            model=NER_MODEL,
            aggregation_strategy="simple",
        )

    return _ner_pipeline


def _clean_cv_header(text: str) -> str:
    """
    Strip emails, URLs, and phone numbers from the header so the NER model
    isn't distracted by non-name tokens.
    """
    text = re.sub(r'\S+@\S+', ' ', text)                          # emails
    text = re.sub(r'http\S+|www\S+|\S*\.com\S+', ' ', text)       # URLs
    text = re.sub(r'\+?\d[\d\s\-]{7,}', ' ', text)                # phone numbers
    return text 


def extract_candidate_name_ner(cv_text: str) -> str:
    """
    Extract the candidate's full name from CV text using NER.

    Strategy:
    - Run NER on the first 200 characters (name is always near the top).
    - Keep only PER entities with at least 2 tokens (first + last name).
    - Score candidates by NER confidence + position (earlier = higher score).
    - Penalise all-lowercase names (likely noise).

    Falls back to "Unknown Candidate" if nothing is found.
    """

    ner = get_ner_pipeline()
    header = _clean_cv_header(cv_text[:200])
    entities = ner(header, aggregation_strategy="simple")

    candidates = []
    for e in entities:
        if e["entity_group"] != "PER":
            continue

        name = e["word"].strip()

        # Require at least two tokens (first + last name minimum)
        if len(name.split()) < 2:
            continue

        score = e.get("score", 0)

        if name.islower():
            score -= 0.2

        score += max(0, (200 - e.get("start", 200)) / 200)

        candidates.append((score, name))


    if not candidates:
        return "Unknown Candidate"

    candidates.sort(reverse=True)
    return candidates[0][1]



_section_classifier_model      = None
_section_classifier_embeddings = None


def _get_section_classifier():
    """
    Load the SentenceTransformer model and precompute all canonical section
    embeddings once. Both are cached as module-level singletons.

    We reuse the same EMBEDDING_MODEL from config so no extra model is loaded.
    """
    global _section_classifier_model, _section_classifier_embeddings

    if _section_classifier_model is None:

        _section_classifier_model = SentenceTransformer(HEADERS_EMBEDDING_MODEL)
        _section_classifier_embeddings = _section_classifier_model.encode(
            KNOWN_SECTIONS,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    return _section_classifier_model, _section_classifier_embeddings



def classify_section(heading: str, threshold: float = 0.80) -> str:
    """
    Map a raw CV heading to the closest canonical section name using
    cosine similarity against KNOWN_SECTIONS.

    Rules:
      - Headings longer than 6 words are unlikely to be section titles → skip.
      - If best cosine similarity >= threshold → return the canonical name
        in Title Case (e.g. "WORK EXPERIENCE" → "Work Experience").
      - If score < threshold → return the original heading unchanged.

    """

    heading_clean = heading.upper().strip()


    if len(heading_clean.split()) > 6:
        return None

    model, section_embeddings = _get_section_classifier()

    heading_embedding = model.encode(
        heading_clean,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    scores     = st_util.cos_sim(heading_embedding, section_embeddings)[0]
    best_score = scores.max().item()
    best_match = KNOWN_SECTIONS[scores.argmax().item()]

    if best_score >= threshold:
        return re.sub(r'[^A-Za-z0-9 ]+', '', heading).strip().title()
    
    return None



def chunk_cvs_with_docling(
    documents_per_cv: List[List[Document]],
) -> List[Document]:
    """
    Docling (header detection) + pdfplumber (text extraction) pipeline.
    Produces exactly ONE chunk per CV section.

    Why two libraries:
      - Docling reliably identifies section_header level=1 elements and their
        text, which we use purely for building the header lookup.
      - pdfplumber extracts text in natural reading order and handles
        multi-column layouts correctly, recovering content that Docling drops
        (e.g. Education section body text).

    Pipeline:
      1. Docling converts the file -> identify all level-1 section headers,
         run through classify_section() -> header_lookup {UPPER: label}.
      2. pdfplumber extracts all text lines from the PDF.
      3. Each line is matched against header_lookup (case-insensitive).
         A match opens a new section bucket.
      4. Non-header lines accumulate as content under the current section.
         Date/location lines misplaced by layout are filtered out.
      5. Candidate name via NER from the first lines of pdfplumber text.
      6. Each section bucket -> one LangChain Document.

    Returns a flat list of all chunk Documents across all CVs.
    """

    converter  = DocumentConverter()
    all_chunks: List[Document] = []

    for cv_docs in documents_per_cv:
        source_cv = cv_docs[0].metadata.get("source_cv", "unknown")

        # ── Step 1: Docling — build header lookup ─────────────────────────────
        result   = converter.convert(source_cv)
        doc_dict = result.document.model_dump(mode="python")
        texts    = doc_dict.get("texts", [])

        header_lookup: dict = {}  # UPPER_TEXT -> title-cased label
        for t in texts:
            label = t.get("label", "")
            level = t.get("level")
            text  = t.get("text", "").strip()
            if not text:
                continue
            try:
                level_int = level.value if hasattr(level, "value") else int(level)
            except (TypeError, ValueError):
                level_int = None
            if label == "section_header" and level_int == 1:
                section_label = classify_section(text)
                if section_label is not None:
                    header_lookup[text.upper()] = section_label

        # ── Step 2: pdfplumber — extract lines in reading order ───────────────
        raw_lines = []
        with pdfplumber.open(source_cv) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if page_text:
                    raw_lines.extend(page_text.splitlines())

        # ── Step 3: Candidate name via NER ────────────────────────────────────
        preview        = " ".join(raw_lines[:30])
        candidate_name = extract_candidate_name_ner(preview)

        # ── Step 4: Split lines into sections ─────────────────────────────────
        sections: dict  = {}
        current_section = "Header & Personal Information"
        sections[current_section] = []

        for line in raw_lines:
            clean = line.strip()
            if not clean:
                continue

            # Check against Docling-detected headers (exact, case-insensitive)
            matched = header_lookup.get(clean.upper())
            if matched:
                current_section = matched
                if current_section not in sections:
                    sections[current_section] = []
                continue

            # Safety net: short ALL-CAPS lines not in lookup
            if clean.isupper() and 1 <= len(clean.split()) <= 5:
                fallback = classify_section(clean)
                if fallback is not None:
                    current_section = fallback
                    if current_section not in sections:
                        sections[current_section] = []
                    continue

            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(clean)

        # ── Step 5: One Document per section ─────────────────────────────────
        cv_chunks = []
        for section_label, lines in sections.items():
            content = "\n".join(lines).strip()
            if not content:
                continue

            cv_chunks.append(Document(
                page_content=(
                    f"Candidate: {candidate_name} | Section: {section_label}\n\n"
                    f"{content}"
                ),
                metadata={
                    "candidate_name": candidate_name,
                    "section":        section_label,
                    "source_cv":      source_cv,
                },
            ))

        all_chunks.extend(cv_chunks)
        print(f"[Docling+pdfplumber] {candidate_name} → {len(cv_chunks)} chunks | "
              f"sections: {[c.metadata['section'] for c in cv_chunks]}")

    return all_chunks