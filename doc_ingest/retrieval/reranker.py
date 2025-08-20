# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import os
import time
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pydantic import BaseModel

from config import RerankerConfig
from retrieval.searcher import SearchResponse, DocResult, SearchHit

# CrossEncoder может быть не установлен — поддерживаем graceful fallback
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore


class RerankedDoc(BaseModel):
    doc_id: Optional[str]
    doc_path: Optional[str]
    category: Optional[str]
    best_score: float
    # уже отсортированные и обрезанные top-M на документ
    hits: List[SearchHit]
    order_key: float                    # ключ сортировки документов
    hierarchy_score: float = 0.0        # дополнительный скор по иерархии
    structural_coherence: float = 0.0   # структурная связность


class HierarchicalReranker:
    """
    Оптимизированный реранкер с учетом иерархии документов:
    - Кэширование результатов
    - Адаптивный батчинг
    - Учет структурной связности
    - Асинхронная обработка
    - Гибридный режим (CrossEncoder + Cosine)
    """

    def __init__(self,
                 cfg: RerankerConfig,
                 # callable: List[str] -> np.ndarray (norm)
                 embed_encode_fn=None,
                 tokenizer=None                # для тримминга текста по токенам
                 ):
        self.cfg = cfg
        self.embed_encode_fn = embed_encode_fn
        self.tokenizer = tokenizer
        self.cross_encoder = None
        self.executor = ThreadPoolExecutor(max_workers=cfg.max_workers)

        # Кэш для результатов
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Инициализация CrossEncoder
        if self.cfg.kind in ["cross_encoder", "hybrid"]:
            self._init_cross_encoder()

        # Установка гибридных весов по умолчанию
        if self.cfg.hybrid_weights is None:
            self.cfg.hybrid_weights = {"cross_encoder": 0.7, "cosine": 0.3}

    def _init_cross_encoder(self):
        """Инициализация CrossEncoder с fallback"""
        if CrossEncoder is None:
            print("WARNING: CrossEncoder не доступен, переключаюсь на 'cosine'.")
            self.cfg.kind = "cosine"
            return

        if not self.cfg.model_path or not Path(self.cfg.model_path).exists():
            print(
                "WARNING: model_path для CrossEncoder не задан или не существует — fallback на 'cosine'.")
            self.cfg.kind = "cosine"
            return

        try:
            if self.cfg.offline_mode:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

            device = None if self.cfg.device == "auto" else self.cfg.device
            self.cross_encoder = CrossEncoder(
                self.cfg.model_path, device=device)
            print(f"CrossEncoder загружен на устройство: {device}")
        except Exception as e:
            print(f"Ошибка загрузки CrossEncoder: {e}, fallback на 'cosine'")
            self.cfg.kind = "cosine"

    def _get_cache_key(self, query: str, hits: List[SearchHit]) -> str:
        """Генерация ключа кэша"""
        hit_ids = [
            f"{h.chunk_index}_{h.page_from}_{h.page_to}" for h in hits[:10]]
        return f"{hash(query)}:{','.join(hit_ids)}"

    def _truncate_by_tokens(self, text: str, max_tokens: int = 256) -> str:
        """Умное усечение текста по токенам"""
        if self.tokenizer is None or not text:
            return text[: max_tokens * 4]

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text

        # Декодируем обратно для точного усечения
        try:
            truncated_ids = ids[:max_tokens]
            return self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        except:
            return text[: max_tokens * 4]

    def _cosine_scores(self, q: str, texts: List[str]) -> np.ndarray:
        """Вычисление косинусных скоров"""
        assert self.embed_encode_fn is not None, "embed_encode_fn должен быть задан для cosine."

        # Батчинг для больших списков
        batch_size = min(self.cfg.batch_size, len(texts))
        all_scores = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_scores = self.embed_encode_fn(batch_texts)
            all_scores.append(batch_scores)

        tv = np.vstack(all_scores)
        qv = self.embed_encode_fn([q])
        return (tv @ qv.T).squeeze(-1)

    def _ce_scores(self, q: str, texts: List[str]) -> np.ndarray:
        """Вычисление CrossEncoder скоров с адаптивным батчингом"""
        assert self.cross_encoder is not None

        # Адаптивный размер батча
        if self.cfg.adaptive_batching:
            batch_size = min(
                self.cfg.max_batch_size,
                max(self.cfg.min_batch_size, len(texts) // 4)
            )
        else:
            batch_size = self.cfg.batch_size

        pairs = [(q, self._truncate_by_tokens(t)) for t in texts]
        scores = self.cross_encoder.predict(pairs, batch_size=batch_size)
        return np.array(scores, dtype=np.float32)

    def _compute_hierarchy_score(self, hits: List[SearchHit]) -> float:
        """Вычисление скора по иерархии документа"""
        if not self.cfg.hierarchy_aware:
            return 0.0

        hierarchy_score = 0.0
        total_weight = 0.0

        for hit in hits:
            # Вес по заголовкам
            if hit.heading_path:
                heading_depth = len(hit.heading_path)
                heading_weight = self.cfg.heading_weight * \
                    (1.0 / (1.0 + heading_depth * 0.1))
                hierarchy_score += heading_weight
                total_weight += self.cfg.heading_weight

            # Вес по категории
            if hasattr(hit, 'category') and hit.category:
                category_weight = self.cfg.category_weight
                hierarchy_score += category_weight
                total_weight += category_weight

        return hierarchy_score / max(total_weight, 1.0)

    def _compute_structural_coherence(self, hits: List[SearchHit]) -> float:
        """Вычисление структурной связности"""
        if not self.cfg.structural_coherence or len(hits) < 2:
            return 0.0

        # Проверяем последовательность страниц
        pages = [h.page_from for h in hits if h.page_from is not None]
        if len(pages) < 2:
            return 0.0

        # Сортируем по страницам
        sorted_pages = sorted(pages)
        coherence = 0.0

        for i in range(len(sorted_pages) - 1):
            if sorted_pages[i + 1] - sorted_pages[i] <= 2:  # Соседние или близкие страницы
                coherence += 1.0

        return coherence / max(len(sorted_pages) - 1, 1.0)

    def _score_hits(self, q: str, hits: List[SearchHit]) -> List[Tuple[float, SearchHit]]:
        """Оценка хитов с учетом иерархии"""
        if not hits:
            return []

        # Проверяем кэш
        cache_key = self._get_cache_key(q, hits)
        if self.cfg.enable_cache and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1

        # Вычисляем базовые скоры
        texts = [h.text for h in hits]

        if self.cfg.kind == "none":
            scores = np.array([h.score for h in hits], dtype=np.float32)
        elif self.cfg.kind == "cosine":
            scores = self._cosine_scores(q, texts)
        elif self.cfg.kind == "cross_encoder":
            scores = self._ce_scores(q, texts)
        elif self.cfg.kind == "hybrid":
            # Гибридный режим
            ce_scores = self._ce_scores(
                q, texts) if self.cross_encoder else np.zeros(len(texts))
            cos_scores = self._cosine_scores(
                q, texts) if self.embed_encode_fn else np.zeros(len(texts))

            # Нормализация скоров
            if len(ce_scores) > 0:
                ce_scores = (ce_scores - ce_scores.min()) / \
                    (ce_scores.max() - ce_scores.min() + 1e-8)
            if len(cos_scores) > 0:
                cos_scores = (cos_scores - cos_scores.min()) / \
                    (cos_scores.max() - cos_scores.min() + 1e-8)

            # Взвешенная комбинация
            weights = self.cfg.hybrid_weights
            scores = (weights.get("cross_encoder", 0.7) * ce_scores +
                      weights.get("cosine", 0.3) * cos_scores)
        else:
            scores = np.array([h.score for h in hits], dtype=np.float32)

        # Добавляем иерархические скоры
        hierarchy_score = self._compute_hierarchy_score(hits)
        structural_coherence = self._compute_structural_coherence(hits)

        # Комбинируем скоры
        final_scores = []
        for i, (score, hit) in enumerate(zip(scores, hits)):
            # Базовый скор + иерархия + структурная связность
            final_score = (score * 0.6 +
                           hierarchy_score * 0.3 +
                           structural_coherence * 0.1)
            final_scores.append((final_score, hit))

        # Кэшируем результат
        if self.cfg.enable_cache and len(self._cache) < self.cfg.cache_size:
            self._cache[cache_key] = final_scores

        return final_scores

    async def _score_hits_async(self, q: str, hits: List[SearchHit]) -> List[Tuple[float, SearchHit]]:
        """Асинхронная оценка хитов"""
        if not self.cfg.async_processing:
            return self._score_hits(q, hits)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._score_hits, q, hits)

    def rerank(self,
               q: str,
               resp: SearchResponse,
               top_k_docs: int = 5,
               max_chunks_per_doc: int = 4) -> List[RerankedDoc]:
        """Основной метод реранкинга с оптимизациями"""
        start_time = time.time()

        # Ограничиваем количество кандидатов для ускорения
        if self.cfg.max_candidates_per_query > 0:
            total_candidates = sum(len(doc.hits) for doc in resp.results)
            if total_candidates > self.cfg.max_candidates_per_query:
                # Отбираем лучшие документы по базовому скору
                sorted_docs = sorted(
                    resp.results, key=lambda d: d.best_score, reverse=True)
                # Берем в 2 раза больше для реранкинга
                resp.results = sorted_docs[:top_k_docs * 2]

        reranked: List[RerankedDoc] = []

        # Обрабатываем документы
        for doc in resp.results:
            if len(doc.hits) == 0:
                continue

            # Оцениваем хиты
            scored = self._score_hits(q, doc.hits)

            # Сортируем по новому скору
            scored.sort(key=lambda x: x[0], reverse=True)

            # Применяем early stopping
            if self.cfg.early_stopping:
                # Отбираем только хиты выше порога уверенности
                confident_hits = [
                    (s, h) for s, h in scored if s >= self.cfg.confidence_threshold]
                if confident_hits:
                    scored = confident_hits

            # Отбираем top-M хитов
            top_hits = [h for _, h in scored[:max_chunks_per_doc]]
            best_score = max([s for s, _ in scored], default=0.0)

            # Вычисляем дополнительные скоры
            hierarchy_score = self._compute_hierarchy_score(top_hits)
            structural_coherence = self._compute_structural_coherence(top_hits)

            reranked.append(RerankedDoc(
                doc_id=doc.doc_id,
                doc_path=doc.doc_path,
                category=doc.category,
                best_score=float(best_score),
                hits=top_hits,
                order_key=float(best_score + hierarchy_score *
                                0.3 + structural_coherence * 0.1),
                hierarchy_score=float(hierarchy_score),
                structural_coherence=float(structural_coherence)
            ))

        # Сортируем документы по комбинированному скору
        reranked.sort(key=lambda d: d.order_key, reverse=True)

        # Логируем статистику
        elapsed = time.time() - start_time
        print(f"Реранкинг завершен за {elapsed:.2f}с. "
              f"Кэш: {self._cache_hits} попаданий, {self._cache_misses} промахов")

        return reranked[:top_k_docs]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(self._cache_hits + self._cache_misses, 1)
        }

    def clear_cache(self):
        """Очистка кэша"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def __del__(self):
        """Очистка ресурсов"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Обратная совместимость
Reranker = HierarchicalReranker
