# CV Intelligence — Vector & Graph RAG Pipeline

> **Transform static resumes into a connected knowledge graph.**
> Powered by Neo4j Graph RAG and Hybrid Vector Search.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Prerequisites](#prerequisites)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Running the Application](#running-the-application)
10. [How It Works](#how-it-works)
    - [Chunking Strategies](#chunking-strategies)
    - [Indexing Pipeline](#indexing-pipeline)
    - [Retrieval Pipeline](#retrieval-pipeline)
    - [Graph Extraction](#graph-extraction)
    - [Answer Generation](#answer-generation)
11. [UI Walkthrough](#ui-walkthrough)
12. [Module Reference](#module-reference)
13. [Configuration Reference](#configuration-reference)
14. [Extending the System](#extending-the-system)
15. [Troubleshooting](#troubleshooting)

---

## Project Overview

**CV Intelligence** is an end-to-end Retrieval-Augmented Generation (RAG) application that ingests PDF and DOCX curriculum vitae files, structures their content into a persistent knowledge graph and a hybrid vector index, and answers natural-language queries about the candidates using a combination of graph traversal and semantic vector search.

The system solves a core HR and talent-intelligence problem: when you have multiple CVs, a plain keyword search or even a simple vector search can miss nuanced relationships (e.g., "Which candidates have worked at companies that use Kubernetes and also hold AWS certifications?"). By fusing **Neo4j Graph RAG** (for structured, relationship-aware retrieval) with **hybrid vector search** (dense cosine + sparse BM25, merged via Reciprocal Rank Fusion), the application can answer both fact-based and semantic queries with high precision.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                           │
│   Upload CVs ──► Select Strategy ──► Build Indexes ──► Ask Questions   │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │         RAG Pipeline (rag_pipeline) │
              └──────┬──────────────────────┬───────┘
                     │                      │
          ┌──────────▼─────────┐  ┌─────────▼──────────┐
          │   CHUNKING MODULE  │  │   GRAPH EXTRACTOR   │
          │  (chunking.py)     │  │  (graph_extractor)  │
          │                    │  │                     │
          │  ┌──────────────┐  │  │  Phase 1: Per-CV    │
          │  │ LLM Chunking │  │  │  extraction         │
          │  └──────────────┘  │  │                     │
          │  ┌──────────────┐  │  │  Phase 2: Cross-doc │
          │  │Docling + NER │  │  │  harmonisation      │
          │  └──────────────┘  │  └─────────┬───────────┘
          └──────────┬─────────┘            │
                     │                      │
          ┌──────────▼─────────┐  ┌─────────▼──────────┐
          │  VECTOR INDEXING   │  │   GRAPH INDEXING    │
          │                    │  │                     │
          │  Qdrant (dense)    │  │   Neo4j (graph DB)  │
          │  BM25 (sparse)     │  │   Nodes + Edges     │
          └──────────┬─────────┘  └─────────┬───────────┘
                     │                      │
              ┌──────▼──────────────────────▼───────┐
              │            QUERY TIME                │
              │                                      │
              │  Leg A: Graph Retriever              │
              │    NL Query → LLM → Cypher → Neo4j   │
              │    → structured graph facts           │
              │                                      │
              │  Leg B: Hybrid Vector Retriever      │
              │    Query → Dense (Qdrant) + Sparse   │
              │    (BM25) → RRF Merge → top-k chunks │
              │                                      │
              │  Answer LLM: graph_facts + chunks    │
              │    → final synthesised answer        │
              └──────────────────────────────────────┘
```

---

## Key Features

**Dual Chunking Strategies**

The system offers two document-parsing approaches selectable at runtime. The first is LLM Chunking, where the full CV text is sent to a GPT model which extracts the candidate's name and splits the content into semantically meaningful sections, returning structured JSON. The second is Docling + NER, a purely local pipeline that combines Docling (for layout-aware section-header detection) with pdfplumber (for accurate text extraction in reading order) and a RoBERTa-large NER model for candidate name extraction — producing more granular chunks without any LLM calls at the parsing stage.

**Hybrid Vector Search**

Three retrieval modes are available at query time: Dense (cosine similarity via Qdrant), Sparse (BM25 keyword matching), and Hybrid. The Hybrid mode runs both retrievers and merges their ranked result lists using Reciprocal Rank Fusion (RRF) with a configurable smoothing constant, achieving the best of both precision and recall.

**Knowledge Graph Construction — Two-Phase LLM Pipeline**

The graph extraction is not a naive single-pass extraction. Phase 1 extracts entities (Person, Skill, Company, Degree, Certification, etc.) and relationships from each CV independently. Phase 2 runs a consistency harmonisation pass that sees the global ontology accumulated from all previously processed CVs and renames labels and relationship types to be consistent, while also discovering cross-document relationships (e.g., two candidates who both worked at the same company).

**Candidate-Aware Retrieval**

The pipeline detects if a query targets a specific candidate by name, filters the search space to only that candidate's chunks, and rewrites the query without the name so that the embedding reflects the topic rather than the identity signal. This dramatically improves precision for candidate-specific questions.

**Read-Only Cypher Safety Guard**

The graph retriever generates Cypher queries via an LLM. Before execution, every generated query is checked against a regex whitelist of write keywords (CREATE, MERGE, SET, DELETE, etc.). Any query containing a write operation is blocked and an empty context is returned, ensuring the graph cannot be modified at query time.

**Persistent Indexes**

Both the Qdrant vector collection and the Neo4j graph persist between application sessions. On startup, the app attempts to reconnect to existing indexes and reconstruct the BM25 index from Qdrant payloads, so previously indexed CVs are immediately available without re-uploading.

---

## Technology Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| LLM (inference) | Azure OpenAI (GPT-4.1-mini) |
| Embeddings | Azure OpenAI (text-embedding-3-small, 1536 dim) |
| Vector Database | Qdrant Cloud |
| Graph Database | Neo4j (Bolt protocol) |
| Sparse Retrieval | BM25 via `rank-bm25` |
| RRF Fusion | Custom implementation |
| Document Parsing | Docling, pdfplumber, PyPDF, Docx2txt |
| NER Model | `Jean-Baptiste/roberta-large-ner-english` (HuggingFace) |
| Section Classifier | `sentence-transformers/all-mpnet-base-v2` |
| LangChain | Chains, Document, OutputParsers |
| Language | Python 3.12+ |

---

## Project Structure

```
Vector&GraphRAG/
│
├── app.py                      # Streamlit UI — entry point
├── rag_pipeline.py             # Main orchestration: indexing + query
├── chunking.py                 # Two chunking strategies (LLM / Docling+NER)
├── graph_extractor.py          # Two-phase knowledge graph extraction
├── embeddings.py               # Azure OpenAI embedding singleton
├── llm.py                      # Azure OpenAI LLM singleton
├── config.py                   # All configuration constants
├── prompts.py                  # All LLM prompt templates (ChatPromptTemplates)
│
├── indexing/
│   ├── qdrant_indexing.py      # Qdrant collection management + dense search
│   └── neo4j_indexing.py       # Neo4j graph write + schema introspection
│
├── retriever/
│   ├── bm25_retriever.py       # BM25 index build + sparse search
│   ├── hybrid_retriever.py     # RRF fusion of dense + sparse results
│   └── graph_retriever.py      # NL → Cypher → Neo4j → formatted facts
│
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

---

## Prerequisites

Before running the application, ensure you have the following services and tools set up.

**Python** version 3.12 or higher is required.

**Azure OpenAI** — You need an Azure OpenAI resource with two deployments: a chat completion deployment (the default model name is `gpt-4.1-mini`) and an embedding deployment (`text-embedding-3-small`). Note down your endpoint URL and API key.

**Qdrant** — A Qdrant instance is required for vector storage. The easiest option is a free [Qdrant Cloud](https://cloud.qdrant.io) cluster. Alternatively, you can run Qdrant locally via Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Neo4j** — A Neo4j instance is required for the knowledge graph. Options include [Neo4j Aura Free](https://neo4j.com/cloud/platform/aura-graph-database/) (managed cloud), or a local instance via Docker:
```bash
docker run \
  --env NEO4J_AUTH=neo4j/your_password \
  -p 7474:7474 -p 7687:7687 \
  neo4j:latest
```

---

## Installation

Clone or extract the project, then install all Python dependencies:

```bash
cd "Vector&GraphRAG"
pip install -r requirements.txt
```

The `requirements.txt` includes the following key packages:
- `langchain`, `langchain-openai`, `langchain-community`, `langchain-huggingface`, `langchain-experimental`
- `streamlit`
- `qdrant-client`
- `neo4j`
- `docling`
- `sentence-transformers`
- `transformers`
- `rank-bm25`
- `pdfplumber`, `pypdf`, `pymupdf`
- `docx2txt`, `unstructured`
- `python-dotenv`

> **Note on model downloads:** On first run with the "Docling + NER" chunking mode, the application will download two HuggingFace models: `Jean-Baptiste/roberta-large-ner-english` (~1.3 GB) and `sentence-transformers/all-mpnet-base-v2` (~420 MB). These are cached locally after the first download.

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Qdrant
QDRANT_URL="https://your-cluster.qdrant.io"
QDRANT_API_KEY="your-qdrant-api-key"

# Azure OpenAI
AZURE_API_KEY="your-azure-openai-api-key"
AZURE_ENDPOINT="https://your-resource.openai.azure.com/"

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

All remaining configuration knobs live in `config.py` and can be tuned without touching the `.env`:

```python
MAX_CVS           = 5       # Maximum number of CVs per session
CHUNKS_PER_QUERY_LLM     = 10   # Top-k chunks returned in LLM-chunking mode
CHUNKS_PER_QUERY_DOCLING = 20   # Top-k chunks returned in Docling mode
RETRIEVAL_MODE    = "Hybrid"    # Default retrieval mode: Sparse / Dense / Hybrid
RRF_K             = 60          # RRF smoothing constant
GRAPH_MAX_RESULTS = 50          # Max rows returned per Neo4j query
EMBEDDING_DIM     = 1536        # Must match the deployed embedding model
```

---

## Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How It Works

### Chunking Strategies

The chunking step transforms raw CV text into structured `Document` objects, each carrying a `candidate_name`, `section`, and `source_cv` in its metadata.

**LLM Chunking** (`chunk_cvs_with_llm`)

All pages of a CV are concatenated into a single text string and sent to the LLM with a structured extraction prompt (`_PARSE_CV_PROMPT`). The LLM returns a JSON object containing:
- `candidate_name`: the extracted full name
- `chunks`: a list of `{section, content}` objects

Each section becomes one `Document`. The chunk text is prefixed with `"Candidate: <name> | Section: <section>"` so that the embedding vector carries both identity and topic signals. If JSON parsing fails, the entire CV is treated as a single "Full CV" chunk.

**Docling + NER** (`chunk_cvs_with_docling`)

This is a local, two-library pipeline that avoids LLM calls during chunking:

1. **Docling** converts the PDF and identifies `section_header` elements at level 1. Each detected header is passed through `classify_section()`, which uses a SentenceTransformer model to compute cosine similarity against a large list of 200+ canonical section names (defined in `config.KNOWN_SECTIONS`). If the similarity exceeds 0.80, the header is mapped to its canonical name; otherwise it is ignored.

2. **pdfplumber** extracts all text lines in natural reading order. Each line is matched against the Docling-built header lookup (case-insensitive). A match opens a new section bucket; all subsequent non-header lines accumulate into that bucket. A safety net also catches short all-caps lines not seen by Docling.

3. **NER** (`extract_candidate_name_ner`) runs `Jean-Baptiste/roberta-large-ner-english` on the first 200 characters of the extracted text. PER-labelled entities with at least two tokens (first + last name) are scored by NER confidence plus a position bonus (earlier in the document = higher score), and the highest-scoring candidate is returned as the name.

### Indexing Pipeline

`create_indexes()` in `rag_pipeline.py` orchestrates the full indexing flow in five steps:

1. **Chunk** all uploaded CVs using the selected strategy.
2. **Build the Qdrant collection** — the collection is deleted and recreated to ensure a clean state. A payload index on `candidate_name` is created for filtered search. All chunk texts are embedded in batches of 20 and upserted as points with full payload (including `page_content`, `candidate_name`, `section`, `source_cv`).
3. **Build the BM25 index** from the same chunk list (in-memory, rebuilt on each session start from Qdrant payloads).
4. **Extract the knowledge graph** using the two-phase LLM pipeline and write it to Neo4j.
5. **Pre-compute and cache the Neo4j schema string** — this is passed into every Cypher-generation prompt to avoid repeated schema introspection.

### Retrieval Pipeline

At query time, `answer_query()` runs two legs in sequence:

**Leg A — Graph Retrieval** (`retrieve_from_graph`)

1. The pre-computed schema string and the user's question are passed to `_GRAPH_CYPHER_PROMPT`, which asks the LLM to write a read-only Cypher query.
2. The raw LLM output is cleaned (markdown fences stripped) and checked against `_WRITE_KEYWORDS` regex. Any query containing write operations is blocked.
3. The safe Cypher is executed against Neo4j. Results are formatted into a human-readable `=== GRAPH FACTS ===` block, rendering Node objects as `(Label prop=value)` and Relationship objects as `[:TYPE prop=value]`.

**Leg B — Hybrid Vector Retrieval** (`_vector_retrieve`)

1. The LLM is asked whether the query targets one or more specific candidates (`_EXTRACT_NAME_PROMPT`). If it does, only that candidate's chunks are eligible for retrieval (`allowed_indices` filtering).
2. If a candidate is identified, the query is rewritten to remove the name (`_REWRITE_QUERY_PROMPT`) so the embedding focuses on the topic, not the person.
3. The `retrieve()` function runs the selected mode:
    - **Dense**: `search_qdrant()` embeds the query and runs a cosine-similarity search, passing a `MatchAny` Qdrant filter for the allowed candidate names if applicable.
    - **Sparse**: `search_bm25()` tokenises the query and scores all eligible chunks using BM25Okapi.
    - **Hybrid**: both are run and their results are merged via `reciprocal_rank_fusion()`.

### Graph Extraction

`build_graph_document()` in `graph_extractor.py` runs a two-phase LLM pipeline:

**Phase 1 — Per-CV Extraction** (`extract_graph_from_chunks`)

Each CV's chunks are assembled into a labelled text block (`[Section]\ncontent`) and sent to `_GRAPH_EXTRACT_PROMPT`. The LLM returns a JSON object with:
- `entities`: list of `{uid, label, name, properties}` objects. UIDs are stable slugs (e.g., `skill_python`, `company_google`). Labels are chosen freely by the LLM (e.g., Skill, Company, Degree, Certification).
- `relationships`: list of `{from_uid, to_uid, type, properties}` objects. Relationship types are uppercase snake-case (e.g., `HAS_SKILL`, `WORKED_AT`, `HOLDS_DEGREE`).

**Phase 2 — Cross-Document Harmonisation** (`harmonise_with_existing`)

After Phase 1, the new extraction is sent to `_GRAPH_CONSISTENCY_PROMPT` alongside the current global ontology summary and the full list of existing entity UIDs and names. The LLM is instructed to:
- Reuse existing node labels and relationship types wherever they are semantically equivalent.
- Introduce new labels/types only if nothing existing fits.
- Identify cross-document relationships (e.g., two candidates who share a skill, worked at the same company, or attended the same university).

The harmonised output is merged into the accumulated `GraphDocument`, deduplicating on UID for entities and on `(from_uid, rel_type, to_uid)` for relationships.

### Answer Generation

The final answer is produced by `_ANSWER_PROMPT`, which receives:
- `graph_context`: the formatted graph facts string from Leg A
- `vector_context`: all retrieved chunk texts joined by separators
- `question`: the original user query

The LLM synthesises these into a single coherent answer.

---

## UI Walkthrough

The Streamlit interface is organised into numbered sections:

**00. Strategy** — Select the chunking approach (LLM Chunking or Docling + NER) before uploading.

**01. Source Control** — Drop up to 5 PDF or DOCX CV files. The file uploader accepts multiple files simultaneously.

**Build Indexes / Clear & Rebuild** — Once files are uploaded, click "⚡ Build Indexes" to run the full indexing pipeline. The button is disabled and shows "✓ Already Indexed" if the current file set and chunking mode match what is already in the databases. "🗑 Clear & Rebuild" wipes both Qdrant and Neo4j and requires a two-click confirmation to prevent accidental data loss.

**Stats Bar** — After indexing, a five-card dashboard shows: Candidates count, Vector Chunks count, Data Segments count, Graph Nodes count, and Graph Edges count. Candidate names are shown as coloured pills below the stats bar.

**Raw Knowledge Base (toggle)** — Expands a filterable view of all chunks, with candidate and section dropdowns. Each chunk is shown in an expandable card with its metadata.

**Graph Schema (toggle)** — Shows the live Neo4j schema string, giving a human-readable view of all node labels, relationship types, and property keys currently in the graph.

**02. Retrieval Engine** — A slider to select the vector retrieval mode: Sparse, Dense, or Hybrid.

**03. Intelligence Prompt** — A text input for natural-language queries (e.g., "Which candidate has the strongest Python background?" or "Find candidates with AWS experience who also have a master's degree").

**04. Graph Facts** — After a query, displays the raw graph facts retrieved from Neo4j by the Cypher leg. Shown in a monospaced, purple-accented code box.

**05. Synthesis** — The final LLM-generated answer, combining graph and vector context.

**06. Vector Evidence** — A two-column grid of chunk cards showing exactly which CV sections were retrieved by the vector leg and contributed to the answer.

---

## Module Reference

### `app.py`
The Streamlit entry point. Manages session state for all index objects (`qdrant_client`, `bm25_index`, `neo4j_driver`, `graph_schema`, `all_chunks`, `k`). Handles the two-click confirmation UX for the destructive clear operation. All index objects are stored in `st.session_state` and survive Streamlit reruns.

### `rag_pipeline.py`
The main orchestration layer. Exposes three public functions: `create_indexes()` for the full build pipeline, `load_existing_indexes()` for session reconnection, `clear_all_indexes()` for destructive wipe, and `answer_query()` for query-time retrieval and answer synthesis.

### `chunking.py`
Implements both chunking strategies. Contains `parse_cv_with_llm()`, `chunk_cvs_with_llm()`, `extract_candidate_name_ner()`, `classify_section()`, and `chunk_cvs_with_docling()`. The NER and SentenceTransformer models are loaded lazily and cached as module-level singletons.

### `graph_extractor.py`
Implements the two-phase graph extraction pipeline. Defines the `GraphEntity`, `GraphRelationship`, and `GraphDocument` dataclasses. Exposes `build_graph_document()` as the public API, which calls `extract_graph_from_chunks()` (Phase 1) and `harmonise_with_existing()` (Phase 2) for each CV.

### `indexing/qdrant_indexing.py`
Manages the Qdrant client singleton and collection lifecycle. Provides `build_qdrant_index()` (wipe + recreate), `add_documents_to_qdrant()` (batch upsert with embeddings), and `search_qdrant()` (cosine-similarity search with optional candidate filter).

### `indexing/neo4j_indexing.py`
Manages the Neo4j driver singleton. Provides `build_neo4j_graph()` (writes all entities and relationships using parameterised Cypher `MERGE` statements), `get_graph_schema()` (introspects the live schema), and `wipe_graph()` (detaches and deletes all nodes).

### `retriever/bm25_retriever.py`
Builds a `BM25Okapi` index from chunk text and provides `search_bm25()` for keyword-based retrieval with optional index filtering.

### `retriever/hybrid_retriever.py`
Provides `reciprocal_rank_fusion()` and the unified `retrieve()` entry point that dispatches to Dense, Sparse, or Hybrid mode.

### `retriever/graph_retriever.py`
Implements the full graph retrieval leg: Cypher generation via LLM, write-keyword safety guard, Neo4j execution, and result formatting.

### `embeddings.py`
Provides a singleton `AzureOpenAIEmbeddings` instance via `get_embeddings()`.

### `llm.py`
Provides a singleton `AzureChatOpenAI` instance via `get_llm()`.

### `config.py`
Single source of truth for all constants: API keys (loaded from `.env`), model names, database connection parameters, pipeline knobs, and the comprehensive `KNOWN_SECTIONS` list (200+ canonical CV section names across all industries).

### `prompts.py`
Contains all LangChain `ChatPromptTemplate` objects used across the system: `_PARSE_CV_PROMPT`, `_GRAPH_EXTRACT_PROMPT`, `_GRAPH_CONSISTENCY_PROMPT`, `_GRAPH_CYPHER_PROMPT`, `_ANSWER_PROMPT`, `_EXTRACT_NAME_PROMPT`, `_REWRITE_QUERY_PROMPT`.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `AZURE_ENDPOINT` | — | Azure OpenAI resource endpoint URL |
| `AZURE_API_KEY` | — | Azure OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model deployment name |
| `EMBEDDING_DIM` | `1536` | Embedding vector dimension |
| `LLM_MODEL` | `gpt-4.1-mini` | Chat completion deployment name |
| `NER_MODEL` | `Jean-Baptiste/roberta-large-ner-english` | HuggingFace NER model for Docling mode |
| `HEADERS_EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | SentenceTransformer for section classification |
| `QDRANT_URL` | — | Qdrant instance URL |
| `QDRANT_API_KEY` | — | Qdrant API key (empty for local) |
| `QDRANT_COLLECTION` | `cv_chunks` | Qdrant collection name |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `MAX_CVS` | `5` | Maximum CVs per session |
| `CHUNKS_PER_QUERY_LLM` | `10` | Top-k for LLM-chunking mode |
| `CHUNKS_PER_QUERY_DOCLING` | `20` | Top-k for Docling mode |
| `RETRIEVAL_MODE` | `Hybrid` | Default retrieval mode |
| `RRF_K` | `60` | RRF smoothing constant |
| `GRAPH_MAX_RESULTS` | `50` | Max Neo4j rows per query |
