# main_embed.py
import json
from pathlib import Path
from typing import List
from tqdm import tqdm

from config import EmbeddingConfig
from models import DocElement, Chunk
from chunking import build_chunks
from embed_faiss import embed_and_build_faiss
from embed_qdrant import push_to_qdrant
from utils_paths import fix_model_paths

# main_embed.py
from embed_utils import load_embedding_stack
from embed_faiss import embed_and_build_faiss
from embed_qdrant import push_to_qdrant


def load_elements(jsonl_path: str) -> List[DocElement]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            items.append(DocElement(**d))
    return items


def group_by_document(elements: List[DocElement]):
    buckets = {}
    for el in elements:
        key = (el.category or "UNK", el.doc_id or el.doc_path or "doc")
        buckets.setdefault(key, []).append(el)
    # сортируем по порядку для стабильности
    for k in buckets:
        buckets[k].sort(key=lambda x: (x.page or 0, x.order or 0))
    return buckets


def run():
    cfg = EmbeddingConfig()
    cfg = fix_model_paths(cfg)  # Автоматически исправляем пути

    Path("outputs/index").mkdir(parents=True, exist_ok=True)

    # 1) грузим модель/токенайзер один раз (офлайн)
    model, tokenizer, prefix = load_embedding_stack(cfg)

    # 2) читаем элементы и группируем
    elements = load_elements("outputs/dataset/elements.jsonl")
    buckets = group_by_document(elements)

    # 3) чанкинг
    all_chunks: List[Chunk] = []
    for (cat, did), els in tqdm(buckets.items(), desc="Chunking per document"):
        ch = build_chunks(els, cfg, model=model,
                          tokenizer=tokenizer, prefix=prefix)
        all_chunks.extend(ch)

    print(f"Chunks total: {len(all_chunks)}")

    # 4) индексы — используем УЖЕ загруженную модель
    if cfg.index_kind in ("faiss", "both"):
        embed_and_build_faiss(all_chunks, cfg, model=model, prefix=prefix)
    if cfg.index_kind in ("qdrant", "both"):
        push_to_qdrant(all_chunks, cfg, model=model, prefix=prefix)


if __name__ == "__main__":
    run()
