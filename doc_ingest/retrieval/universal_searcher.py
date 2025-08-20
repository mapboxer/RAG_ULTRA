# retrieval/universal_searcher.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import numpy as np
import time
from functools import lru_cache

from retrieval.searcher import SemanticSearcher, DocResult, SearchResponse, SearchHit
from retrieval.reranker import Reranker, RerankedDoc
from retrieval.graph_searcher import GraphSearcher, GraphSearchResult
from ..entity_extractor import EntityExtractor, ExtractedEntities

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None  # type: ignore


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("ё", "е").split())


def _tok_ru(s: str) -> List[str]:
    import re
    return [t for t in re.split(r"[^a-zA-Zа-яА-Я0-9]+", _norm(s)) if t]


@dataclass
class Weights:
    dense: float = 1.0
    heading: float = 0.7
    path: float = 0.8
    category: float = 0.3
    bm25: float = 0.5


@dataclass
class Gating:
    enable: bool = True
    idf_percentile: float = 0.85   # какое IDF считать «редким» (0..1)
    # штраф за отсутствие редких токенов в path/heading (0..1)
    penalty: float = 0.7


@dataclass
class UniversalSearchConfig:
    semantic_top_k: int = 180
    oversample: int = 5
    window: int = 1
    top_k_docs: int = 5
    max_chunks_per_doc: int = 4
    weights: Weights = Weights()
    bm25_enable: bool = True
    gating: Gating = Gating()
    # пути к индексам полей/лексикону
    doc_fields_meta: str = "outputs/index/doc_fields_meta.json"
    doc_fields_npz: str = "outputs/index/doc_fields.npz"
    lexicon_path: str = "outputs/index/lexicon_doc_fields.json"

    # Оптимизации производительности
    enable_cache: bool = True
    cache_size: int = 1000
    batch_processing: bool = True
    max_batch_size: int = 50

    # Интеграция с графом и извлечением сущностей
    use_graph_search: bool = True
    use_entity_extraction: bool = True
    graph_weight: float = 0.3  # Вес результатов из графа
    entity_weight: float = 0.2  # Вес совпадения по сущностям


class UniversalSearcher:
    """
    Универсальный ранжировщик без ручных правил:
    - мультиполевые плотные признаки (heading/path/category) + dense из ретривера
    - опц. BM25 по навигационным полям
    - IDF-гейтинг по «редким» токенам запроса (в path/heading)
    - дедупликация хитов внутри документа
    """

    def __init__(self,
                 base_searcher: SemanticSearcher,
                 cfg: Optional[UniversalSearchConfig] = None,
                 final_reranker: Optional[Reranker] = None,
                 graph_searcher: Optional[GraphSearcher] = None,
                 entity_extractor: Optional[EntityExtractor] = None):
        self.base = base_searcher
        self.cfg = cfg or UniversalSearchConfig()
        self.final_reranker = final_reranker

        # doc-level поля
        self._load_doc_fields()

        # Кэш для результатов
        self._cache = {}
        self._cache_hits = 0

        # Инициализация графа и экстрактора сущностей
        self.graph_searcher = graph_searcher
        self.entity_extractor = entity_extractor

        if self.cfg.use_graph_search and not self.graph_searcher:
            try:
                self.graph_searcher = GraphSearcher()
            except Exception as e:
                print(f"Не удалось инициализировать GraphSearcher: {e}")
                self.cfg.use_graph_search = False

        if self.cfg.use_entity_extraction and not self.entity_extractor:
            try:
                self.entity_extractor = EntityExtractor(use_llm=False)
            except Exception as e:
                print(f"Не удалось инициализировать EntityExtractor: {e}")
                self.cfg.use_entity_extraction = False
        self._cache_misses = 0

    def _load_doc_fields(self):
        """Загрузка полей документов с оптимизацией"""
        try:
            J = json.loads(
                Path(self.cfg.doc_fields_meta).read_text(encoding="utf-8"))
            self.doc_keys: List[str] = J["doc_keys"]
            self.doc_meta: List[Dict[str, Any]] = J["meta"]

            Z = np.load(self.cfg.doc_fields_npz)
            self.H = Z["heading"].astype("float32")
            self.P = Z["path"].astype("float32")
            self.C = Z["category"].astype("float32")
            self.dim = self.H.shape[1]
            self.key2pos: Dict[str, int] = {
                k: i for i, k in enumerate(self.doc_keys)}

            # заранее токенизируем навигационные поля (для гейтинга/BM25)
            self.doc_nav_tokens: List[List[str]] = []
            for m in self.doc_meta:
                s = (m.get("path_text") or "") + " " + \
                    (m.get("heading_text") or "")
                self.doc_nav_tokens.append(_tok_ru(s))

            # лексикон для IDF-гейтинга
            if self.cfg.gating.enable and Path(self.cfg.lexicon_path).exists():
                L = json.loads(
                    Path(self.cfg.lexicon_path).read_text(encoding="utf-8"))
                self.N_lex = int(L["N"])
                self.df_map: Dict[str, int] = {
                    k: int(v) for k, v in L["df"].items()}
            else:
                self.N_lex, self.df_map = 0, {}

        except Exception as e:
            print(f"Ошибка загрузки полей документов: {e}")
            # Fallback значения
            self.doc_keys = []
            self.doc_meta = []
            self.H = np.array([])
            self.P = np.array([])
            self.C = np.array([])
            self.dim = 0
            self.key2pos = {}
            self.doc_nav_tokens = []
            self.N_lex, self.df_map = 0, {}

    def _get_cache_key(self, query: str, top_k: int, **kwargs) -> str:
        """Генерация ключа кэша"""
        params = f"k{top_k}_w{self.cfg.window}"
        return f"{hash(query)}:{params}"

    @lru_cache(maxsize=1000)
    def _encode_query_fields(self, query: str) -> Dict[str, np.ndarray]:
        """Кэшированное кодирование полей запроса"""
        if not hasattr(self.base, 'model'):
            return {}

        try:
            # Кодируем запрос для разных полей
            query_encoded = self.base.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]

            return {
                "query": query_encoded,
                "query_tokens": _tok_ru(query)
            }
        except Exception as e:
            print(f"Ошибка кодирования запроса: {e}")
            return {}

    def _compute_field_scores(self, query_encoded: np.ndarray, doc_pos: int) -> Dict[str, float]:
        """Вычисление скоров по полям документа"""
        if doc_pos >= len(self.H):
            return {"heading": 0.0, "path": 0.0, "category": 0.0}

        # Косинусное сходство по полям
        heading_score = float(np.dot(self.H[doc_pos], query_encoded))
        path_score = float(np.dot(self.P[doc_pos], query_encoded))
        category_score = float(np.dot(self.C[doc_pos], query_encoded))

        return {
            "heading": heading_score,
            "path": path_score,
            "category": category_score
        }

    def _compute_bm25_score(self, query_tokens: List[str], doc_pos: int) -> float:
        """Вычисление BM25 скора"""
        if not self.cfg.bm25_enable or doc_pos >= len(self.doc_nav_tokens):
            return 0.0

        try:
            if not hasattr(self, '_bm25_index'):
                # Создаем BM25 индекс
                all_tokens = [
                    tokens for tokens in self.doc_nav_tokens if tokens]
                if all_tokens:
                    self._bm25_index = BM25Okapi(all_tokens)
                else:
                    return 0.0

            doc_tokens = self.doc_nav_tokens[doc_pos]
            if not doc_tokens:
                return 0.0

            score = self._bm25_index.get_scores(query_tokens)
            return float(score[doc_pos]) if doc_pos < len(score) else 0.0

        except Exception as e:
            print(f"Ошибка BM25: {e}")
            return 0.0

    def _compute_gating_penalty(self, query_tokens: List[str], doc_pos: int) -> float:
        """Вычисление штрафа за отсутствие редких токенов"""
        if not self.cfg.gating.enable or not self.df_map:
            return 1.0

        try:
            doc_tokens = self.doc_nav_tokens[doc_pos] if doc_pos < len(
                self.doc_nav_tokens) else []
            if not doc_tokens:
                return self.cfg.gating.penalty

            # Находим редкие токены запроса
            rare_tokens = []
            for token in query_tokens:
                df = self.df_map.get(token, 0)
                if df > 0 and df / self.N_lex <= (1 - self.cfg.gating.idf_percentile):
                    rare_tokens.append(token)

            if not rare_tokens:
                return 1.0

            # Проверяем наличие редких токенов в документе
            found_rare = sum(1 for token in rare_tokens if token in doc_tokens)
            coverage = found_rare / len(rare_tokens)

            # Применяем штраф
            return 1.0 - (1.0 - coverage) * (1.0 - self.cfg.gating.penalty)

        except Exception as e:
            print(f"Ошибка гейтинга: {e}")
            return 1.0

    def _combine_scores(self,
                        dense_score: float,
                        field_scores: Dict[str, float],
                        bm25_score: float,
                        gating_penalty: float) -> float:
        """Комбинирование всех скоров"""
        weights = self.cfg.weights

        # Нормализуем BM25 скор
        bm25_norm = _minmax(np.array([bm25_score]))[
            0] if bm25_score != 0 else 0.0

        # Комбинируем скоры
        combined = (weights.dense * dense_score +
                    weights.heading * field_scores["heading"] +
                    weights.path * field_scores["path"] +
                    weights.category * field_scores["category"] +
                    weights.bm25 * bm25_norm)

        # Применяем гейтинг
        final_score = combined * gating_penalty

        return float(final_score)

    def search(self, q: str, **kwargs) -> SearchResponse:
        """Основной метод поиска с оптимизациями и интеграцией графа"""
        start_time = time.time()

        # Проверяем кэш
        cache_key = self._get_cache_key(q, self.cfg.top_k_docs, **kwargs)
        if self.cfg.enable_cache and cache_key in self._cache:
            self._cache_hits += 1
            cached_result = self._cache[cache_key]
            cached_result["took_ms"] = int((time.time() - start_time) * 1000)
            return SearchResponse(**cached_result)

        self._cache_misses += 1

        # Извлекаем сущности из запроса если включено
        extracted_entities = None
        if self.cfg.use_entity_extraction and self.entity_extractor:
            try:
                extracted_entities = self.entity_extractor.extract(q)
            except Exception as e:
                print(f"Ошибка извлечения сущностей: {e}")

        # Кодируем запрос
        query_fields = self._encode_query_fields(q)
        if not query_fields:
            return SearchResponse(
                query=q, took_ms=0, total_candidates=0, results=[]
            )

        query_encoded = query_fields["query"]
        query_tokens = query_fields["query_tokens"]

        # Базовый поиск
        base_response = self.base.search_raw(
            q,
            top_k=self.cfg.semantic_top_k,
            oversample=self.cfg.oversample
        )

        # Поиск по графу если включен
        graph_results = []
        if self.cfg.use_graph_search and self.graph_searcher:
            try:
                graph_results = self.graph_searcher.search(
                    q,
                    top_k=self.cfg.top_k_docs * 2,
                    use_entities=True
                )
            except Exception as e:
                print(f"Ошибка поиска по графу: {e}")

        # Обрабатываем результаты
        enhanced_results = []
        doc_scores_map = {}  # Для объединения скоров из разных источников

        for doc_result in base_response.results:
            enhanced_hits = []
            doc_key = f"{doc_result.category}::{doc_result.doc_path}"

            for hit in doc_result.hits:
                # Находим позицию документа в полях
                doc_pos = self.key2pos.get(doc_key, -1)

                if doc_pos >= 0:
                    # Вычисляем дополнительные скоры
                    field_scores = self._compute_field_scores(
                        query_encoded, doc_pos)
                    bm25_score = self._compute_bm25_score(
                        query_tokens, doc_pos)
                    gating_penalty = self._compute_gating_penalty(
                        query_tokens, doc_pos)

                    # Комбинируем скоры
                    enhanced_score = self._combine_scores(
                        hit.score, field_scores, bm25_score, gating_penalty
                    )

                    # Добавляем бонус за совпадение сущностей
                    if extracted_entities:
                        entity_bonus = self._calculate_entity_bonus(
                            doc_key, extracted_entities
                        )
                        enhanced_score += entity_bonus * self.cfg.entity_weight

                    # Обновляем хит
                    hit.score = enhanced_score

                enhanced_hits.append(hit)

            # Сортируем хиты по новому скору
            enhanced_hits.sort(key=lambda h: h.score, reverse=True)

            # Обновляем документ
            doc_result.hits = enhanced_hits
            doc_result.best_score = enhanced_hits[0].score if enhanced_hits else 0.0

            # Сохраняем в мапу для объединения с графом
            doc_scores_map[doc_key] = doc_result

            enhanced_results.append(doc_result)

        # Добавляем результаты из графа
        if graph_results:
            enhanced_results = self._merge_graph_results(
                enhanced_results, graph_results, doc_scores_map
            )

        # Сортируем документы по лучшему скору
        enhanced_results.sort(key=lambda d: d.best_score, reverse=True)

        # Применяем финальный реранкер если есть
        if self.final_reranker:
            try:
                reranked = self.final_reranker.rerank(
                    q,
                    SearchResponse(
                        query=q, took_ms=0, total_candidates=len(enhanced_results),
                        results=enhanced_results
                    ),
                    top_k_docs=self.cfg.top_k_docs,
                    max_chunks_per_doc=self.cfg.max_chunks_per_doc
                )

                # Конвертируем обратно в DocResult
                final_results = []
                for rd in reranked:
                    final_results.append(DocResult(
                        doc_id=rd.doc_id,
                        doc_path=rd.doc_path,
                        category=rd.category,
                        best_score=rd.best_score,
                        hits=rd.hits,
                        context="\n\n".join([h.text for h in rd.hits]),
                        page_span=None,  # Можно вычислить из хитов
                        heading_preview=None  # Можно вычислить из хитов
                    ))

                enhanced_results = final_results

            except Exception as e:
                print(f"Ошибка реранкинга: {e}")

        # Ограничиваем количество результатов
        final_results = enhanced_results[:self.cfg.top_k_docs]

        # Собираем финальный ответ
        response_data = {
            "query": q,
            "took_ms": 0,  # Будет установлено ниже
            "total_candidates": len(base_response.results),
            "results": final_results
        }

        # Кэшируем результат
        if self.cfg.enable_cache and len(self._cache) < self.cfg.cache_size:
            self._cache[cache_key] = response_data.copy()

        # Устанавливаем время выполнения
        elapsed = time.time() - start_time
        response_data["took_ms"] = int(elapsed * 1000)

        return SearchResponse(**response_data)

    def _calculate_entity_bonus(self, doc_key: str, entities: ExtractedEntities) -> float:
        """Вычисляет бонус за совпадение сущностей"""
        try:
            # Получаем метаданные документа
            doc_pos = self.key2pos.get(doc_key, -1)
            if doc_pos < 0:
                return 0.0

            bonus = 0.0

            # Проверяем совпадение по домену
            if entities.domain and self.doc_categories:
                doc_cat = self.doc_categories[doc_pos] if doc_pos < len(
                    self.doc_categories) else ""
                if entities.domain.value.lower() in doc_cat.lower():
                    bonus += 0.3

            # Проверяем совпадение по ключевым словам
            if entities.object_attributes and self.doc_headings:
                doc_heading = self.doc_headings[doc_pos] if doc_pos < len(
                    self.doc_headings) else ""
                for attr in entities.object_attributes:
                    if attr.lower() in doc_heading.lower():
                        bonus += 0.2
                        break

            return min(bonus, 1.0)

        except Exception as e:
            print(f"Ошибка вычисления entity bonus: {e}")
            return 0.0

    def _merge_graph_results(self,
                             semantic_results: List[DocResult],
                             graph_results: List[GraphSearchResult],
                             doc_scores_map: Dict[str, DocResult]) -> List[DocResult]:
        """Объединяет результаты семантического поиска и поиска по графу"""
        try:
            # Добавляем бонусы из графа к существующим документам
            for graph_result in graph_results:
                # Формируем ключ документа из графа
                if graph_result.path:
                    # Пытаемся найти соответствие в семантических результатах
                    for doc_key, doc_result in doc_scores_map.items():
                        if graph_result.path in doc_key or doc_key in graph_result.path:
                            # Добавляем бонус от графа
                            graph_bonus = graph_result.score * self.cfg.graph_weight
                            doc_result.best_score += graph_bonus
                            break
                    else:
                        # Если документ найден только в графе, создаем новый DocResult
                        if graph_result.content:
                            new_doc = DocResult(
                                doc_id=graph_result.node_id,
                                doc_path=graph_result.path or graph_result.node_id,
                                category="graph",
                                best_score=graph_result.score * self.cfg.graph_weight,
                                hits=[SearchHit(
                                    chunk_id=0,
                                    score=graph_result.score,
                                    text=graph_result.content[:500] if graph_result.content else "",
                                    metadata=graph_result.metadata or {}
                                )],
                                context=graph_result.content[:1000] if graph_result.content else ""
                            )
                            semantic_results.append(new_doc)

            return semantic_results

        except Exception as e:
            print(f"Ошибка объединения результатов графа: {e}")
            return semantic_results

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
        self._encode_query_fields.cache_clear()
