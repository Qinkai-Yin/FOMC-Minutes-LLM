# app/rag_cli.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

INDEX_DIR = Path("index")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"

# ---------- load index ----------
def load_index():
    emb = np.load(INDEX_DIR / "embeddings.npy")
    data = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return emb, data["chunks"], data["metas"]

# ---------- retrieval ----------
def cosine_topk(qv: np.ndarray, mat: np.ndarray, k: int = 5):
    sims = mat @ qv  # both normalized
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

# ---------- generator (load once) ----------
_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
_mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
_gen = pipeline("text2text-generation", model=_mdl, tokenizer=_tok)

def build_answer(context_chunks: List[str], question: str) -> str:
    """
    Add chunks one by one until reaching the token budget to avoid 512-limit of flan-t5-small.
    """
    MAX_IN_TOKENS = 480   # leave room for output; 480 is safe for 512 context window
    sep = "\n\n---\n\n"
    header = ("You are a helpful assistant answering questions about recent FOMC minutes. "
              "Use ONLY the provided context. If the answer is not in the context, say you don't know.\n\n")
    q = f"Question: {question}\n\n"

    chosen = []
    for ch in context_chunks:
        tentative = header + q + "Context:\n" + sep.join(chosen + [ch]) + "\n\nAnswer:"
        if len(_tok(tentative).input_ids) <= MAX_IN_TOKENS:
            chosen.append(ch)
        else:
            break

    if not chosen and context_chunks:
        # if even first chunk is too long, truncate hard
        ids = _tok(header + q + "Context:\n" + context_chunks[0] + "\n\nAnswer:",
                   truncation=True, max_length=MAX_IN_TOKENS).input_ids
        prompt = _tok.decode(ids, skip_special_tokens=True)
    else:
        prompt = header + q + "Context:\n" + sep.join(chosen) + "\n\nAnswer:"

    out = _gen(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]
    return out.strip()

def main():
    ap = argparse.ArgumentParser(description="RAG CLI over FOMC minutes (local).")
    ap.add_argument("--q", "--question", dest="question", required=True, help="your question")
    ap.add_argument("--k", type=int, default=5, help="top-k chunks to retrieve")
    ap.add_argument("--only", type=str, default="", help="only search sources containing this substring")
    args = ap.parse_args()

    emb, chunks, metas = load_index()

    # optional pre-filter by source (url or local path)
    if args.only:
        keep = []
        for i, m in enumerate(metas):
            src = m.get("source_url") or m.get("source_text")
            if args.only in src:
                keep.append(i)
        if keep:
            emb = emb[keep]
            chunks = [chunks[i] for i in keep]
            metas  = [metas[i]  for i in keep]
        else:
            print(f"[warn] no sources matched '--only {args.only}'; using all sources")

    enc = SentenceTransformer(EMB_MODEL)
    qv = enc.encode([args.question], normalize_embeddings=True)[0].astype("float32")

    idxs, sims = cosine_topk(qv, emb, k=args.k)
    picked = [chunks[i] for i in idxs]

    # ---------- generate ----------
    answer = build_answer(picked, args.question)

    print("\n[Answer]")
    print(answer)

    print("\n[Sources]")
    for i, j in enumerate(idxs):
        meta = metas[j]
        src = meta.get("source_url") or meta.get("source_text")
        print(f"{i+1}. {src}  (sim={sims[i]:.3f})")

if __name__ == "__main__":
    main()
