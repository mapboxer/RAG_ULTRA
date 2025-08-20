# embed_faiss.py
import json
from pathlib import Path
import numpy as np
import faiss
from typing import List
from models import Chunk
from config import EmbeddingConfig
from sentence_transformers import SentenceTransformer

def embed_and_build_faiss(chunks: List[Chunk], cfg: EmbeddingConfig,
                          model: SentenceTransformer, prefix: str = ""):
    Path("outputs/index").mkdir(parents=True, exist_ok=True)

    texts = [ (prefix + c.text) for c in chunks ]
    vecs = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=128, show_progress_bar=True
    )
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(vecs)
    faiss.write_index(index, cfg.faiss_out)

    with open(cfg.meta_out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "id": c.id,
                "category": c.category,
                "doc_id": c.doc_id,
                "doc_path": c.doc_path,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "token_len": c.token_len,
                "heading_path": c.heading_path,
                "page_from": c.page_from,
                "page_to": c.page_to,
                "element_types": c.element_types,
                "media_refs": c.media_refs
            }, ensure_ascii=False) + "\n")
    print(f"FAISS index -> {cfg.faiss_out}, meta -> {cfg.meta_out}")