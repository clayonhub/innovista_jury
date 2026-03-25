"""
build_embeddings.py — Offline Faculty Embedding Builder (MiniLM Edition)
========================================================================
Run this ONCE before starting the Streamlit app:

    python build_embeddings.py

What it does:
  1. Reads faculty_master_list.csv
  2. Encodes all faculty using all-MiniLM-L6-v2 (ultra-fast, ~80MB)
  3. Saves:
       faculty_minilm_embeddings.npy  ← embedding matrix (384-dim)
       faculty_minilm_meta.json       ← faculty metadata + csv hash

After running this, Streamlit will load from disk instantly
and will NOT need to re-embed anything at startup.
The small MiniLM model is loaded at runtime for fast query embedding.
"""

import os
import json
import hashlib
import time
import warnings

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Config (must match app.py) ────────────────────────────────────────────────
CSV_PATH   = "faculty_master_list.csv"
EMB_PATH   = "faculty_bge_embeddings.npy"
META_PATH  = "faculty_bge_meta.json"
BATCH_SIZE = 128  # Lower batch size for the 1.3GB model on Colab GPU
EMB_MODEL  = "BAAI/bge-large-en-v1.5"

REQUIRED_COLS = [
    "name", "college", "department", "designation",
    "research_interest", "email", "phone",
    "research_work", "consulting", "profile_url",
]

META_COLS = [
    "name", "college", "department", "designation",
    "email", "phone", "profile_url", "consulting", "research_interest",
]


def chunk_text(text: str, max_words=100) -> list:
    parts = [p.strip() for p in text.split('|') if p.strip()]
    final_chunks = []
    
    for part in parts:
        if len(part.split()) <= max_words:
            final_chunks.append(part)
        else:
            # Smart Fallback: Split by sentences so we never chop a thought in half
            sentences = [s.strip() + "." for s in part.split('. ') if s.strip()]
            current_chunk = []
            current_words = 0
            
            for sentence in sentences:
                sent_words = len(sentence.split())
                # If adding this sentence pushes us over the limit, save the chunk and start a new one
                if current_words + sent_words > max_words and current_chunk:
                    final_chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_words = sent_words
                else:
                    current_chunk.append(sentence)
                    current_words += sent_words
                    
            # Add any remaining sentences as the final chunk
            if current_chunk:
                final_chunks.append(" ".join(current_chunk))
                
    return final_chunks

def build_doc_chunks(row) -> list:
    """Break research text into multiple small, dense chunks to prevent dilution."""
    ri = str(row.get("research_interest", "")).strip()
    rw = str(row.get("research_work", "")).strip()
    cns = str(row.get("consulting", "")).strip()
    
    chunks = []
    if ri:
        for c in chunk_text(ri, 100):
            chunks.append(f"Research Interests: {c}")
    if rw:
        for c in chunk_text(rw, 100):
            chunks.append(f"Research Work: {c}")
    if cns:
        for c in chunk_text(cns, 100):
            chunks.append(f"Consulting: {c}")
            
    if not chunks:
        chunks.append("No research data.")
    return chunks


def df_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(
            df[["name", "research_interest", "research_work"]], index=False
        ).values.tobytes()
    ).hexdigest()[:16]


def main():
    # ── 1. Load CSV ──────────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"❌  '{CSV_PATH}' not found. Place it in the same directory.")
        return

    print(f"📂  Loading {CSV_PATH} …")
    t0 = time.time()
    df = pd.read_csv(CSV_PATH, usecols=lambda c: c in REQUIRED_COLS, dtype=str, low_memory=False)
    df = df[[c for c in REQUIRED_COLS if c in df.columns]].copy()
    df = df.dropna(subset=["name"]).fillna("").reset_index(drop=True)
    print(f"    {len(df):,} faculty rows loaded in {time.time()-t0:.1f}s")

    # ── 2. Check if embeddings are already up to date ─────────────────────────
    csv_hash = df_hash(df)
    chunk_map_path = "faculty_bge_chunk_map.npy"
    if os.path.exists(EMB_PATH) and os.path.exists(META_PATH) and os.path.exists(chunk_map_path):
        with open(META_PATH, encoding="utf-8") as f:
            stored = json.load(f)
        if stored.get("csv_hash") == csv_hash and stored.get("n_rows") == len(df):
            print("✅  Embeddings are already up to date. Nothing to do.")
            print(f"    EMB : {EMB_PATH}  ({os.path.getsize(EMB_PATH)/1e6:.1f} MB)")
            print(f"    META: {META_PATH}")
            return

    # ── 3. Load model ─────────────────────────────────────────────────────────
    print(f"\n🤖  Loading model: {EMB_MODEL} …")
    t1 = time.time()
    model = SentenceTransformer(EMB_MODEL)
    print(f"    Model loaded in {time.time()-t1:.1f}s")

    # ── 4. Build texts ────────────────────────────────────────────────────────
    print(f"\n📝  Building document chunks for {len(df):,} faculty …")
    all_chunks = []
    chunk_to_fac = []
    
    for fac_idx, row in df.iterrows():
        chunks = build_doc_chunks(row)
        all_chunks.extend(chunks)
        chunk_to_fac.extend([fac_idx] * len(chunks))
        
    print(f"    Generated {len(all_chunks):,} total chunks.")

    # ── 5. Encode ─────────────────────────────────────────────────────────────
    print(f"⚡  Encoding with batch_size={BATCH_SIZE} — this should be very fast …")
    t2 = time.time()
    emb = model.encode(
        all_chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Ensure L2-normed
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    print(f"    Encoded {len(emb):,} vectors in {time.time()-t2:.1f}s  shape={emb.shape}")

    # ── 6. Save embeddings ────────────────────────────────────────────────────
    print(f"\n💾  Saving {EMB_PATH} …")
    np.save(EMB_PATH, emb)
    
    print(f"💾  Saving chunk map to {chunk_map_path} …")
    np.save(chunk_map_path, np.array(chunk_to_fac, dtype=np.int32))

    # ── 7. Save metadata ──────────────────────────────────────────────────────
    print(f"💾  Saving {META_PATH} …")
    meta_cols = [c for c in META_COLS if c in df.columns]
    meta = df[meta_cols].to_dict(orient="records")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"csv_hash": csv_hash, "n_rows": len(df), "rows": meta},
            f,
            ensure_ascii=False,
        )

    size_mb = os.path.getsize(EMB_PATH) / 1e6
    print(f"\n✅  Done in {time.time()-t0:.1f}s total")
    print(f"    EMB      : {EMB_PATH}  ({size_mb:.1f} MB)")
    print(f"    CHUNK MAP: {chunk_map_path} ({len(chunk_to_fac):,} mapped chunks)")
    print(f"    META     : {META_PATH}  ({len(meta):,} records)")
    print(f"\n🚀  You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
