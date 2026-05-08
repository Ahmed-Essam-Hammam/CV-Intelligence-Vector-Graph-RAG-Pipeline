import streamlit as st
import tempfile
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from chunking import chunk_cvs_with_llm, chunk_cvs_with_docling
from langchain_core.documents import Document as LCDoc
from rag_pipeline import create_indexes, load_existing_indexes, clear_all_indexes, answer_query
from config import MAX_CVS, RETRIEVAL_MODE

st.set_page_config(
    page_title="CV Intelligence",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;600&display=swap');

:root {
    --bg-main: #05070a;
    --bg-card: #0d1117;
    --accent: #00ffa3;
    --accent-secondary: #00d4ff;
    --accent-graph: #a78bfa;
    --text-main: #e6edf3;
    --text-dim: #8b949e;
    --border: #30363d;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-main);
    color: var(--text-main);
}
.stApp { background-color: var(--bg-main); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 5rem; max-width: 1400px; }

.hero {
    padding: 3rem;
    background: linear-gradient(135deg, rgba(0, 255, 163, 0.05) 0%, rgba(167, 139, 250, 0.05) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    margin-bottom: 3rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    background: linear-gradient(to bottom, var(--accent), var(--accent-graph));
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.3em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 4rem;
    line-height: 1.1;
    color: #ffffff;
    margin: 0 0 1rem;
}
.hero-title em {
    font-style: normal;
    background: linear-gradient(90deg, var(--accent), var(--accent-graph));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.1rem;
    color: var(--text-dim);
    max-width: 650px;
    line-height: 1.6;
}

.stats-bar {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1.5rem;
    margin: 2rem 0;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    transition: transform 0.3s ease;
}
.stat-card:hover { transform: translateY(-5px); border-color: var(--accent); }
.stat-card.graph-stat:hover { border-color: var(--accent-graph); }
.stat-value {
    display: block;
    font-size: 2rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 0.25rem;
}
.stat-value.graph { color: var(--accent-graph); }
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.section-dot {
    width: 8px;
    height: 8px;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent);
    border-radius: 50%;
}
.section-dot.graph {
    background: var(--accent-graph);
    box-shadow: 0 0 10px var(--accent-graph);
}
.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--text-main);
}

.candidate-pill {
    background: rgba(0, 255, 163, 0.1);
    border: 1px solid rgba(0, 255, 163, 0.3);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    color: var(--accent);
    font-family: 'DM Mono', monospace;
    display: inline-block;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.chunk-card {
    background: #12171f;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
}
.chunk-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent-secondary);
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: space-between;
}
.chunk-text {
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.6;
}

.graph-facts-box {
    background: #0f0a1e;
    border: 1px solid rgba(167, 139, 250, 0.4);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #c4b5fd;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-word;
}
.graph-facts-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--accent-graph);
    margin-bottom: 0.75rem;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 2px dashed var(--border);
    padding: 2rem;
    border-radius: 15px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-graph)) !important;
    color: black !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 0.75rem !important;
    border-radius: 8px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stTextInput input {
    background-color: #0d1117 !important;
    border: 1px solid var(--border) !important;
    color: white !important;
    padding: 1rem !important;
}
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Artificial Intelligence · Graph + Hybrid RAG</div>
    <h1 class="hero-title">CV <em>Intelligence</em></h1>
    <p class="hero-sub">
        Transform static resumes into a connected knowledge graph.
        Powered by Neo4j Graph RAG and Hybrid Vector Search.
    </p>
</div>
""", unsafe_allow_html=True)

# ── 00. Strategy ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><div class="section-dot"></div>'
    '<div class="section-title">00. Strategy</div></div>',
    unsafe_allow_html=True,
)
chunking_mode = st.radio(
    "Select chunking approach",
    options=["LLM Chunking", "Docling + NER"],
    horizontal=True,
    label_visibility="collapsed",
)

# ── 01. Upload ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><div class="section-dot"></div>'
    '<div class="section-title">01. Source Control</div></div>',
    unsafe_allow_html=True,
)
uploaded_files = st.file_uploader(
    f"Drop up to {MAX_CVS} PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files and len(uploaded_files) > MAX_CVS:
    st.warning(f"Maximum {MAX_CVS} CVs allowed — current count: {len(uploaded_files)}.")
    st.stop()

# ── Index management via session_state ───────────────────────────────────────
#
# We store the indexes in st.session_state so they survive reruns but can be
# explicitly invalidated by the "Clear & Rebuild" button.
#
# State keys:
#   indexes_ready  : bool — True once indexes are loaded/built
#   qdrant_client  : QdrantClient
#   bm25_index     : BM25Okapi
#   neo4j_driver   : neo4j.Driver
#   graph_schema   : str
#   all_chunks     : List[Document]
#   k              : int
#   indexed_files  : frozenset of filenames that are currently indexed
#   indexed_mode   : chunking mode used when indexes were built

def _prepare_documents(files, mode):
    """Load uploaded files into Document lists, return (documents_per_cv, tmp_paths)."""
    documents_per_cv = []
    tmp_paths_to_keep = []
    for f in files:
        name   = f.name
        data   = f.read()
        suffix = Path(name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        if mode == "Docling + NER":
            placeholder = LCDoc(page_content="", metadata={"source_cv": tmp_path})
            documents_per_cv.append([placeholder])
            tmp_paths_to_keep.append(tmp_path)
        else:
            try:
                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                elif suffix == ".docx":
                    loader = Docx2txtLoader(tmp_path)
                else:
                    os.unlink(tmp_path)
                    continue
                docs = loader.load()
                for d in docs:
                    d.metadata["source_cv"] = name
                documents_per_cv.append(docs)
            finally:
                os.unlink(tmp_path)
    return documents_per_cv, tmp_paths_to_keep


# ── Try to load existing indexes on first run ─────────────────────────────────
if "indexes_ready" not in st.session_state:
    st.session_state.indexes_ready = False

if not st.session_state.indexes_ready:
    with st.spinner("Connecting to existing indexes..."):
        try:
            (
                st.session_state.qdrant_client,
                st.session_state.bm25_index,
                st.session_state.neo4j_driver,
                st.session_state.graph_schema,
                st.session_state.all_chunks,
                st.session_state.k,
            ) = load_existing_indexes(chunking_mode)

            if st.session_state.all_chunks:
                st.session_state.indexes_ready  = True
                st.session_state.indexed_mode   = chunking_mode
                st.session_state.indexed_files  = frozenset(
                    d.metadata.get("source_cv", "") for d in st.session_state.all_chunks
                )
                st.success(
                    f"Loaded existing indexes — "
                    f"{len(st.session_state.all_chunks)} chunks, "
                    f"ready to query."
                )
            else:
                st.info("No existing data found. Upload CVs and click **Build Indexes**.")
        except Exception as e:
            st.info(f"No existing indexes found ({e}). Upload CVs and click **Build Indexes**.")

# ── Control bar: Build / Clear & Rebuild ─────────────────────────────────────
if uploaded_files:
    current_files = frozenset(f.name for f in uploaded_files)
    already_indexed = (
        st.session_state.indexes_ready
        and st.session_state.get("indexed_files") == current_files
        and st.session_state.get("indexed_mode") == chunking_mode
    )

    btn_col1, btn_col2 = st.columns([1, 1])

    with btn_col1:
        build_label = "✓ Already Indexed" if already_indexed else "⚡ Build Indexes"
        build_clicked = st.button(build_label, disabled=already_indexed)

    with btn_col2:
        # Two-click confirmation for destructive clear
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False

        if not st.session_state.confirm_clear:
            if st.button("🗑 Clear & Rebuild", disabled=not st.session_state.indexes_ready):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("This will wipe all indexes. Are you sure?")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Yes, wipe everything"):
                    with st.spinner("Clearing Qdrant + Neo4j..."):
                        clear_all_indexes()
                    st.session_state.indexes_ready  = False
                    st.session_state.confirm_clear  = False
                    st.session_state.pop("indexed_files", None)
                    st.session_state.pop("indexed_mode",  None)
                    st.rerun()
            with cc2:
                if st.button("Cancel"):
                    st.session_state.confirm_clear = False
                    st.rerun()

    if build_clicked and not already_indexed:
        documents_per_cv, tmp_paths = _prepare_documents(uploaded_files, chunking_mode)
        with st.spinner("Building knowledge graph and vector indexes — this may take a few minutes..."):
            (
                st.session_state.qdrant_client,
                st.session_state.bm25_index,
                st.session_state.neo4j_driver,
                st.session_state.graph_schema,
                st.session_state.all_chunks,
                st.session_state.k,
            ) = create_indexes(documents_per_cv, chunking_mode=chunking_mode)
        for p in tmp_paths:
            if os.path.exists(p):
                os.unlink(p)
        st.session_state.indexes_ready = True
        st.session_state.indexed_files = frozenset(f.name for f in uploaded_files)
        st.session_state.indexed_mode  = chunking_mode
        st.success("Indexes built successfully!")
        st.rerun()

# ── Gate: nothing to show until indexes are ready ────────────────────────────
if not st.session_state.indexes_ready or not st.session_state.all_chunks:
    st.stop()

# Convenience aliases from session_state
qdrant_client = st.session_state.qdrant_client
bm25_index    = st.session_state.bm25_index
neo4j_driver  = st.session_state.neo4j_driver
graph_schema  = st.session_state.graph_schema
all_chunks    = st.session_state.all_chunks
k             = st.session_state.k

candidate_names = list(dict.fromkeys(
    d.metadata.get("candidate_name", "Unknown") for d in all_chunks
))
sections_all = list(dict.fromkeys(
    d.metadata.get("section", "Unknown") for d in all_chunks
))

# Pull graph stats from Neo4j for the stats bar
def get_graph_stats(driver):
    try:
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rel_count  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return node_count, rel_count
    except Exception:
        return 0, 0

node_count, rel_count = get_graph_stats(neo4j_driver)

# ── Stats ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-card">
        <span class="stat-value">{len(candidate_names)}</span>
        <span class="stat-label">Candidates</span>
    </div>
    <div class="stat-card">
        <span class="stat-value">{len(all_chunks)}</span>
        <span class="stat-label">Vector Chunks</span>
    </div>
    <div class="stat-card">
        <span class="stat-value">{len(sections_all)}</span>
        <span class="stat-label">Data Segments</span>
    </div>
    <div class="stat-card graph-stat">
        <span class="stat-value graph">{node_count}</span>
        <span class="stat-label">Graph Nodes</span>
    </div>
    <div class="stat-card graph-stat">
        <span class="stat-value graph">{rel_count}</span>
        <span class="stat-label">Graph Edges</span>
    </div>
</div>
""", unsafe_allow_html=True)

pills_html = "".join(f'<span class="candidate-pill">{n}</span>' for n in candidate_names)
st.markdown(f'<div style="margin-bottom: 2rem;">{pills_html}</div>', unsafe_allow_html=True)

# ── Knowledge Base viewer ─────────────────────────────────────────────────────
show_kb = st.toggle("Access Raw Knowledge Base")
if show_kb:
    st.markdown(
        '<div class="section-header"><div class="section-dot"></div>'
        '<div class="section-title">Raw Data Nodes</div></div>',
        unsafe_allow_html=True,
    )
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        selected_candidate = st.selectbox("Candidate Filter", ["All"] + sorted(candidate_names))
    with fcol2:
        selected_section = st.selectbox("Section Filter", ["All"] + sorted(sections_all))

    filtered = [
        d for d in all_chunks
        if (selected_candidate == "All" or d.metadata.get("candidate_name") == selected_candidate)
        and (selected_section == "All" or d.metadata.get("section") == selected_section)
    ]

    for i, doc in enumerate(filtered, 1):
        with st.expander(
            f"Chunk #{i:02d} — "
            f"{doc.metadata.get('candidate_name', 'Unknown')}  ·  "
            f"{doc.metadata.get('section', 'Unknown')}"
        ):
            st.markdown(
                f'<div class="chunk-meta">'
                f'<span>{doc.metadata.get("section", "General")}</span>'
                f'<span>{doc.metadata.get("source_cv", "")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="chunk-text">{doc.page_content}</div>',
                unsafe_allow_html=True,
            )

# ── Graph Schema viewer ───────────────────────────────────────────────────────
show_schema = st.toggle("Inspect Graph Schema")
if show_schema:
    st.markdown(
        '<div class="section-header">'
        '<div class="section-dot graph"></div>'
        '<div class="section-title">Live Graph Schema</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="graph-facts-box">{graph_schema}</div>',
        unsafe_allow_html=True,
    )

# ── 02. Retrieval Engine ──────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><div class="section-dot"></div>'
    '<div class="section-title">02. Retrieval Engine</div></div>',
    unsafe_allow_html=True,
)
retrieval_mode = st.select_slider(
    "Select Vector Precision Mode",
    options=["Sparse", "Dense", "Hybrid"],
    value=RETRIEVAL_MODE,
)

# ── 03. Query ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><div class="section-dot"></div>'
    '<div class="section-title">03. Intelligence Prompt</div></div>',
    unsafe_allow_html=True,
)
user_query = st.text_input(
    "Query",
    placeholder="e.g., Which candidate has the strongest Python background?",
    label_visibility="collapsed",
)

if user_query:
    with st.spinner("Running graph + vector retrieval..."):
        answer, graph_context, vector_docs = answer_query(
            qdrant_client=qdrant_client,
            bm25_index=bm25_index,
            neo4j_driver=neo4j_driver,
            graph_schema=graph_schema,
            all_chunks=all_chunks,
            user_query=user_query,
            k=k,
            retrieval_mode=retrieval_mode,
        )

    # ── 04. Graph Facts ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">'
        '<div class="section-dot graph"></div>'
        '<div class="section-title">04. Graph Facts</div></div>',
        unsafe_allow_html=True,
    )
    if graph_context and graph_context.strip():
        st.markdown(
            f'<div class="graph-facts-box">{graph_context}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No graph facts retrieved for this query.")

    # ── 05. Answer ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">'
        '<div class="section-dot" style="background:var(--accent-secondary); '
        'box-shadow: 0 0 10px var(--accent-secondary);"></div>'
        '<div class="section-title">05. Synthesis</div></div>',
        unsafe_allow_html=True,
    )
    st.write(answer)

    # ── 06. Vector Evidence ───────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><div class="section-dot"></div>'
        '<div class="section-title">06. Vector Evidence</div></div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    for i, doc in enumerate(vector_docs):
        card_html = f"""
        <div class="chunk-card">
            <div class="chunk-meta">
                <span style="color:var(--accent)">{doc.metadata.get("candidate_name", "Unknown")}</span>
                <span>{doc.metadata.get("section", "Section")}</span>
            </div>
            <div class="chunk-text">{doc.page_content}</div>
        </div>
        """
        if i % 2 == 0:
            col1.markdown(card_html, unsafe_allow_html=True)
        else:
            col2.markdown(card_html, unsafe_allow_html=True)