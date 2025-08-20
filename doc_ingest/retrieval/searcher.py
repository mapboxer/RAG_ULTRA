# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from functools import lru_cache
import time

import faiss
import numpy as np
from pydantic import BaseModel

from config import EmbeddingConfig
from embed_utils import load_embedding_stack


# ---------- Pydantic модели ----------

class SearchHit(BaseModel):
    score: float
    chunk_index: int
    text: str
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    heading_path: Optional[List[str]] = None
    element_types: Optional[List[str]] = None
    media_refs: Optional[List[str]] = None


class DocResult(BaseModel):
    doc_id: Optional[str] = None
    doc_path: Optional[str] = None
    category: Optional[str] = None
    best_score: float
    hits: List[SearchHit]
    context: str
    page_span: Optional[Tuple[Optional[int], Optional[int]]] = None
    heading_preview: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    took_ms: int
    total_candidates: int
    results: List[DocResult]
    debug: Optional[Dict[str, Any]] = None  # опционально


# ---------- utils ----------

def _doc_key(meta: Dict[str, Any]) -> str:
    cat = meta.get("category") or "UNK"
    did = meta.get("doc_id") or meta.get("doc_path") or "doc"
    return f"{cat}::{did}"


# ---------- Поисковик ----------

class SemanticSearcher:
    """
    Импортируемый семантический поисковик (без HTTP).
    Работает поверх локальных эмбеддингов, FAISS и meta.jsonl.
    """

    def __init__(self, cfg: Optional[EmbeddingConfig] = None):
        self.cfg = cfg or EmbeddingConfig()
        self.model, self.tokenizer, self.prefix = load_embedding_stack(
            self.cfg)

        # FAISS
        if not Path(self.cfg.faiss_out).exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.cfg.faiss_out}")
        self.index: faiss.Index = faiss.read_index(self.cfg.faiss_out)

        # META
        if not Path(self.cfg.meta_out).exists():
            raise FileNotFoundError(
                f"META jsonl not found: {self.cfg.meta_out}")

        self.meta: List[Dict[str, Any]] = []
        with open(self.cfg.meta_out, "r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

        if self.index.ntotal != len(self.meta):
            raise RuntimeError(
                f"FAISS vectors ({self.index.ntotal}) and META rows ({len(self.meta)}) mismatch. "
                f"Rebuild index/meta to align."
            )

        # Маппинги по документам
        self.doc2vecidx: Dict[str, List[int]] = {}
        self.vecidx2dockey: List[str] = [""] * len(self.meta)

        for i, m in enumerate(self.meta):
            dk = _doc_key(m)
            self.doc2vecidx.setdefault(dk, []).append(i)
            self.vecidx2dockey[i] = dk

        for dk, idxs in self.doc2vecidx.items():
            idxs.sort(key=lambda ix: (self.meta[ix].get("from_order") or 0,
                                      self.meta[ix].get("to_order") or 0))

    # --- низкоуровневые утилиты ---

    @lru_cache(maxsize=1000)
    def _encode_query(self, text: str) -> np.ndarray:
        """Кэшированное кодирование запроса"""
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Батчевое кодирование с оптимизацией"""
        if not texts:
            return np.array([])

        # Используем конфигурируемый размер батча
        batch_size = self.cfg.batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings)

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])

    def _search_faiss(self, query_vec: np.ndarray, top_k: int, oversample: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Оптимизированный поиск в FAISS"""
        # Увеличиваем top_k для oversampling
        search_k = min(top_k * oversample, self.index.ntotal)

        # Выполняем поиск
        scores, indices = self.index.search(query_vec.reshape(1, -1), search_k)

        return scores[0], indices[0]

    # --- отладочный "сырые топ-k" ---

    def search_raw(self,
                   q: str,
                   top_k: int = 10,
                   oversample: int = 5,
                   categories: Optional[Set[str]] = None,
                   doc_ids: Optional[Set[str]] = None,
                   doc_paths: Optional[Set[str]] = None,
                   heading_prefixes: Optional[List[str]] = None,
                   min_score: Optional[float] = None) -> SearchResponse:
        """
        Сырой поиск по FAISS без группировки по документам.
        Возвращает top-k чанков с их скором.
        """
        start_time = time.time()

        # Кодируем запрос
        query_vec = self._encode_query(q)

        # Поиск в FAISS
        scores, indices = self._search_faiss(query_vec, top_k, oversample)

        # Фильтрация и сбор результатов
        hits: List[SearchHit] = []
        total_candidates = 0

        for score, idx in zip(scores, indices):
            if idx < 0:  # FAISS может вернуть -1 для пустых результатов
                continue

            meta = self.meta[idx]
            total_candidates += 1

            # Применяем фильтры
            if not self._filter_mask(meta, categories, doc_ids, doc_paths, heading_prefixes, min_score, score):
                continue

            # Создаем хит
            hit = SearchHit(
                score=float(score),
                chunk_index=int(meta.get("chunk_index", idx)),
                text=meta.get("text") or "",
                page_from=meta.get("page_from"),
                page_to=meta.get("page_to"),
                heading_path=meta.get("heading_path") or None,
                element_types=meta.get("element_types") or None,
                media_refs=meta.get("media_refs") or None
            )
            hits.append(hit)

            if len(hits) >= top_k:
                break

        # Группируем по документам для совместимости
        doc_results = self._group_hits_by_docs(hits)

        elapsed = time.time() - start_time
        return SearchResponse(
            query=q,
            took_ms=int(elapsed * 1000),
            total_candidates=total_candidates,
            results=doc_results
        )

    def _group_hits_by_docs(self, hits: List[SearchHit]) -> List[DocResult]:
        """Группировка хитов по документам"""
        doc_groups: Dict[str, List[SearchHit]] = {}

        for hit in hits:
            # Находим документ для этого хита
            doc_key = None
            for dk, vec_indices in self.doc2vecidx.items():
                if any(self.meta[vi].get("chunk_index") == hit.chunk_index for vi in vec_indices):
                    doc_key = dk
                    break

            if doc_key:
                if doc_key not in doc_groups:
                    doc_groups[doc_key] = []
                doc_groups[doc_key].append(hit)

        # Создаем DocResult для каждого документа
        results = []
        for doc_key, doc_hits in doc_groups.items():
            # Сортируем хиты по скору
            doc_hits.sort(key=lambda h: h.score, reverse=True)

            # Собираем контекст
            context_parts = []
            pages = []
            for hit in doc_hits:
                if hit.text.strip():
                    context_parts.append(hit.text)
                if hit.page_from is not None:
                    pages.append(hit.page_from)
                if hit.page_to is not None:
                    pages.append(hit.page_to)

            context = "\n\n".join(context_parts)
            page_span = (min(pages) if pages else None,
                         max(pages) if pages else None)

            # Preview заголовков
            heading_preview = None
            if doc_hits and doc_hits[0].heading_path:
                heading_preview = " > ".join(doc_hits[0].heading_path[:2])

            # Парсим doc_key для получения категории и пути
            if "::" in doc_key:
                category, doc_path = doc_key.split("::", 1)
            else:
                category, doc_path = "UNK", doc_key

            results.append(DocResult(
                doc_id=doc_key,
                doc_path=doc_path,
                category=category,
                best_score=float(doc_hits[0].score),
                hits=doc_hits,
                context=context,
                page_span=page_span,
                heading_preview=heading_preview
            ))

        # Сортируем по лучшему скору
        results.sort(key=lambda d: d.best_score, reverse=True)
        return results

    @staticmethod
    def _filter_mask(
        meta: Dict[str, Any],
        categories: Optional[Set[str]],
        doc_ids: Optional[Set[str]],
        doc_paths: Optional[Set[str]],
        heading_prefixes: Optional[List[str]],
        min_score: Optional[float],
        score: float,
    ) -> bool:
        """Фильтрация результатов поиска"""
        # category
        if categories is not None:
            cat = meta.get("category")
            if not cat or cat not in categories:
                return False

        # doc_id
        if doc_ids is not None:
            did = meta.get("doc_id")
            if not did or did not in doc_ids:
                return False

        # doc_path
        if doc_paths is not None:
            dpath = meta.get("doc_path")
            if not dpath or dpath not in doc_paths:
                return False

        # heading_prefix
        if heading_prefixes:
            hp = meta.get("heading_path") or []
            ok = False
            for prefix in heading_prefixes:
                if any(h.startswith(prefix) for h in hp):
                    ok = True
                    break
            if not ok:
                return False

        # min_score
        if min_score is not None and score < min_score:
            return False

        return True

    def _gather_doc_context(
        self,
        doc_key: str,
        center_vec_idx: int,
        window: int
    ) -> Tuple[List[int], str, Tuple[Optional[int], Optional[int]], Optional[str]]:
        """Сбор контекста документа вокруг центрального чанка"""
        doc_vecs = self.doc2vecidx.get(doc_key, [])
        if not doc_vecs:
            m = self.meta[center_vec_idx]
            return [center_vec_idx], (m.get("text") or ""), (m.get("page_from"), m.get("page_to")), None

        try:
            pos = doc_vecs.index(center_vec_idx)
        except ValueError:
            m = self.meta[center_vec_idx]
            return [center_vec_idx], (m.get("text") or ""), (m.get("page_from"), m.get("page_to")), None

        start = max(0, pos - window)
        end = min(len(doc_vecs), pos + window + 1)
        span = doc_vecs[start:end]

        parts, pages = [], []
        for ix in span:
            mm = self.meta[ix]
            t = mm.get("text") or ""
            if t.strip():
                parts.append(t)
            if mm.get("page_from") is not None:
                pages.append(mm["page_from"])
            if mm.get("page_to") is not None:
                pages.append(mm["page_to"])
        context_text = "\n\n".join(parts)
        page_span = (min(pages) if pages else None,
                     max(pages) if pages else None)

        hp = self.meta[center_vec_idx].get("heading_path") or []
        heading_preview = " > ".join(hp[:2]) if hp else None

        return span, context_text, page_span, heading_preview

    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        return {
            "query_cache_info": self._encode_query.cache_info(),
            "total_meta_entries": len(self.meta),
            "total_documents": len(self.doc2vecidx),
            "faiss_index_size": self.index.ntotal
        }

    def clear_cache(self):
        """Очистка кэша"""
        self._encode_query.cache_clear()
