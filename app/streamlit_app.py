# app/streamlit_app.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

INDEX_DIR = Path("index")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"   # runs on CPU; switch to flan-t5-base if you want stronger generation

# ------------------------- cached loaders -------------------------
@st.cache_resource
def load_index() -> Tuple[np.ndarray, List[str], List[dict]]:
    emb = np.load(INDEX_DIR / "embeddings.npy")
    data = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return emb, data["chunks"], data["metas"]

@st.cache_resource
def load_retriever() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL)

@st.cache_resource
def load_generator():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok)
    return tok, gen

# ------------------------- retrieval utils -------------------------
def cosine_topk(qv: np.ndarray, mat: np.ndarray, k: int = 5):
    sims = mat @ qv  # embeddings are normalized
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def build_answer(chunks: List[str], question: str) -> str:
    """Greedily add chunks until reaching a token budget (flan-t5-small ~512)."""
    tok, gen = load_generator()
    MAX_IN_TOKENS = 480  # leave some room for the prompt

    header = (
        "You are a helpful assistant answering questions about recent FOMC minutes. "
        "Use ONLY the provided context. If the answer is not in the context, say you don't know.\n\n"
    )
    q = f"Question: {question}\n\n"
    sep = "\n\n---\n\n"

    chosen = []
    for ch in chunks:
        tentative = header + q + "Context:\n" + sep.join(chosen + [ch]) + "\n\nAnswer:"
        if len(tok(tentative).input_ids) <= MAX_IN_TOKENS:
            chosen.append(ch)
        else:
            break

    if not chosen and chunks:
        ids = tok(
            header + q + "Context:\n" + chunks[0] + "\n\nAnswer:",
            truncation=True,
            max_length=MAX_IN_TOKENS,
        ).input_ids
        prompt = tok.decode(ids, skip_special_tokens=True)
    else:
        prompt = header + q + "Context:\n" + sep.join(chosen) + "\n\nAnswer:"

    out = gen(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]
    return out.strip()

# ------------------------- UI -------------------------
st.set_page_config(page_title="FOMC Minutes — Mini RAG", page_icon="📄", layout="wide")
st.title("📄 FOMC Minutes — Mini RAG")

with st.sidebar:
    st.subheader("Settings")
    k = st.slider("Top-k chunks", 1, 5, 2)
    only = st.text_input("Filter sources (URL substring or date)", value="")
    st.caption("e.g., `20250730` or `fomcminutes20250730`. Leave empty for no filter.")
    st.divider()
    st.subheader("Index status")
    if not (INDEX_DIR / "embeddings.npy").exists():
        st.error("Index not found. Build it first in the terminal: `python app/build_index.py`")
    else:
        st.success("Index loaded.")

question = st.text_area(
    "Ask a question about the minutes",
    height=90,
    placeholder="e.g., In the July 30, 2025 minutes, what did the Committee say about inflation risks?",
)
ask = st.button("Ask")

if ask and question.strip():
    try:
        emb, chunks, metas = load_index()

        # optional pre-filter by source
        key = only.strip()
        if key:
            keep = []
            for i, m in enumerate(metas):
                src = m.get("source_url") or m.get("source_text")
                if key in src:
                    keep.append(i)
            if keep:
                emb = emb[keep]
                chunks = [chunks[i] for i in keep]
                metas  = [metas[i]  for i in keep]
            else:
                st.warning(f"No sources matched: `{key}`. Using all sources.")

        retriever = load_retriever()
        qv = retriever.encode([question], normalize_embeddings=True)[0].astype("float32")
        idxs, sims = cosine_topk(qv, emb, k=k)
        picked = [chunks[i] for i in idxs]

        answer = build_answer(picked, question)

        st.markdown("### ✅ Answer")
        st.write(answer)

        st.markdown("### 🔎 Sources")
        for i, j in enumerate(idxs):
            meta = metas[j]
            src = meta.get("source_url") or meta.get("source_text")
            st.markdown(f"**{i+1}.** [{src}]({src}) — *(sim={sims[i]:.3f})*")

        with st.expander("Show retrieved chunks"):
            for i, j in enumerate(idxs):
                st.markdown(f"**Chunk {i+1}**  *(sim={sims[i]:.3f})*")
                st.write(chunks[j])
                st.markdown("---")
    except FileNotFoundError:
        st.error("Index not built. Run `python app/build_index.py` in the terminal.")
