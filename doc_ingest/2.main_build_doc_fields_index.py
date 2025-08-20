# tools/build_doc_fields_index.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from config import EmbeddingConfig
from embed_utils import load_embedding_stack
from utils_paths import fix_model_paths

ELEMENTS = "outputs/dataset/elements.jsonl"
OUT_META = "outputs/index/doc_fields_meta.json"
OUT_NPZ = "outputs/index/doc_fields.npz"


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _path_to_text(p: str) -> str:
    p = (p or "").replace("\\", "/")
    tokens = re.split(r"[\/_\-\.\(\)]", p)
    tokens = [t for t in tokens if t and not t.isdigit()]
    return _norm_space(" ".join(tokens))


def _uniq_keep_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _doc_key(cat: str | None, doc_id: str | None, doc_path: str | None) -> str:
    base = doc_id or doc_path or "doc"
    return f"{(cat or 'UNK')}::{base}"


def run():
    Path("outputs/index").mkdir(parents=True, exist_ok=True)

    # 1) Сбор заголовков по документам
    by_doc: Dict[str, Dict[str, Any]] = {}
    headings_by_doc: Dict[str, List[str]] = defaultdict(list)

    with open(ELEMENTS, "r", encoding="utf-8") as f:
        for line in f:
            el = json.loads(line)
            cat = el.get("category")
            did = el.get("doc_id")
            dpath = el.get("doc_path")
            dk = _doc_key(cat, did, dpath)
            by_doc.setdefault(
                dk, {"category": cat, "doc_id": did, "doc_path": dpath})
            hp = el.get("heading_path") or []
            if hp:
                headings_by_doc[dk].extend(hp)

    for dk, lst in list(headings_by_doc.items()):
        headings_by_doc[dk] = _uniq_keep_order([_norm_space(h) for h in lst])

        # 2) Локальная embed-модель
    cfg = EmbeddingConfig()
    cfg = fix_model_paths(cfg)  # Автоматически исправляем пути
    model, _, prefix = load_embedding_stack(cfg)

    def enc(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
        X = model.encode([prefix + t for t in texts],
                         convert_to_numpy=True, normalize_embeddings=True,
                         batch_size=128, show_progress_bar=True)
        return X.astype("float32")

    # 3) Подготовка текстов полей
    doc_keys = list(by_doc.keys())

    heading_texts = []
    path_texts = []
    cat_texts = []
    for dk in doc_keys:
        meta = by_doc[dk]
        hs = headings_by_doc.get(dk, [])
        htext = " \n".join(hs) if hs else ""
        ptext = _path_to_text(meta.get("doc_path") or "")
        ctext = _norm_space(meta.get("category") or "")
        heading_texts.append(htext)
        path_texts.append(ptext)
        cat_texts.append(ctext)

    # 4) Эмбеддинги полей
    H = enc(heading_texts)
    P = enc(path_texts)
    C = enc(cat_texts)

    def _l2n(X: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return (X / n).astype("float32")

    H, P, C = _l2n(H), _l2n(P), _l2n(C)

    # 5) Сохраняем мета + эмбеддинги
    meta_out = []
    for dk, htext, ptext in zip(doc_keys, heading_texts, path_texts):
        m = by_doc[dk]
        meta_out.append({
            "doc_key": dk,
            "category": m.get("category"),
            "doc_id": m.get("doc_id"),
            "doc_path": m.get("doc_path"),
            "heading_text": htext,
            "path_text": ptext
        })

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({"doc_keys": doc_keys, "meta": meta_out},
                  f, ensure_ascii=False, indent=2)

    np.savez_compressed(OUT_NPZ, heading=H, path=P, category=C)
    print(f"OK: saved {len(doc_keys)} docs -> {OUT_META}, {OUT_NPZ}")


if __name__ == "__main__":
    run()
