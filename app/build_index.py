# app/build_index.py
"""
Build an embedding index from data/text/*.txt using a local Sentence-Transformers model.
Outputs:
- index/chunks.json     : chunk texts + minimal metadata
- index/embeddings.npy  : float32 matrix [num_chunks, dim]
"""

from __future__ import annotations
import glob, json, re
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from pathlib import Path
import json as _json
META_MAP = {}
meta_file = Path("index/meta.json")
if meta_file.exists():
    try:
        for item in _json.loads(meta_file.read_text(encoding="utf-8")):
            META_MAP[item.get("text_path", "")] = item.get("source", "")
    except Exception:
        pass

TEXT_DIR = Path("data/text")
INDEX_DIR = Path("index"); INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, ~384-dim

def chunk_text(s: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    s = re.sub(r"\s+\n", "\n", s).strip()
    chunks = []
    start = 0
    while start < len(s):
        end = min(len(s), start + size)
        cut = s[start:end]
        if end < len(s):
            window = s[max(start, end - 120):end]
            m = re.search(r"[.!?]\s", window)
            if m:
                end = max(start, end - (len(window) - m.end()))
                cut = s[start:end]
        chunks.append(cut.strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]

def main():
    files = sorted(glob.glob(str(TEXT_DIR / "*.txt")))
    if not files:
        print("[warn] no text files under data/text/. Run fetch_minutes.py first.")
        return

    model = SentenceTransformer(EMB_MODEL)
    all_chunks: List[str] = []
    metas: List[Dict] = []

    for fp in files:
        txt = Path(fp).read_text(encoding="utf-8", errors="ignore")
        pieces = chunk_text(txt)
        for i, chunk in enumerate(pieces):
            metas.append({"source_text": fp, "source_url": META_MAP.get(fp, ""),"chunk_id": i, "preview": chunk[:200]})
            all_chunks.append(chunk)

    if not all_chunks:
        print("[warn] no chunks produced.")
        return

    print(f"[info] embedding {len(all_chunks)} chunks with {EMB_MODEL} ...")
    X = model.encode(all_chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    X = X.astype("float32")

    np.save(INDEX_DIR / "embeddings.npy", X)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps({"chunks": all_chunks, "metas": metas}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("[ok] wrote index/embeddings.npy and index/chunks.json")

if __name__ == "__main__":
    main()
