"""
Модуль для поиска по графу документов с учетом извлеченных сущностей
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass
import networkx as nx

from ..graph_builder import HierarchicalGraphBuilder
from ..entity_extractor import EntityExtractor, ExtractedEntities

logger = logging.getLogger(__name__)


@dataclass
class GraphSearchResult:
    """Результат поиска по графу"""
    node_id: str
    node_type: str  # document, section, chunk
    score: float
    path: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    importance: float = 0.5
    related_nodes: List[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "score": self.score,
            "path": self.path,
            "title": self.title,
            # Ограничиваем для превью
            "content": self.content[:500] if self.content else None,
            "entities": self.entities,
            "importance": self.importance,
            "related_nodes": self.related_nodes[:5] if self.related_nodes else [],
            "metadata": self.metadata
        }


class GraphSearcher:
    """
    Поисковик по графу документов с учетом:
    - Извлеченных сущностей из запроса
    - Структурной иерархии документов
    - Семантических связей между узлами
    - Важности узлов (PageRank)
    """

    def __init__(self,
                 graph_path: str = "outputs/graphs/corpus_graph",
                 entity_extractor: Optional[EntityExtractor] = None,
                 use_pagerank: bool = True):
        """
        Args:
            graph_path: Путь к сохраненному графу
            entity_extractor: Экстрактор сущностей
            use_pagerank: Использовать ли PageRank для ранжирования
        """
        self.graph_builder = HierarchicalGraphBuilder()
        self.graph_loaded = self.graph_builder.load_graph(
            Path(graph_path).stem)

        if not self.graph_loaded:
            logger.warning("Граф не загружен, создаем пустой граф")
            self.graph_builder.graph = nx.DiGraph()

        self.entity_extractor = entity_extractor or EntityExtractor(
            use_llm=False)
        self.use_pagerank = use_pagerank

        # Вычисляем PageRank если нужно
        if self.use_pagerank and len(self.graph_builder.graph) > 0:
            try:
                self.pagerank_scores = nx.pagerank(
                    self.graph_builder.graph, alpha=0.85)
            except:
                self.pagerank_scores = {}
                logger.warning("Не удалось вычислить PageRank")
        else:
            self.pagerank_scores = {}

    def search(self,
               query: str,
               top_k: int = 10,
               use_entities: bool = True,
               expand_graph: bool = True,
               filters: Optional[Dict[str, Any]] = None) -> List[GraphSearchResult]:
        """
        Поиск по графу документов

        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            use_entities: Использовать ли извлечение сущностей
            expand_graph: Расширять ли поиск на связанные узлы
            filters: Фильтры (domain, doc_type, date_range и т.д.)

        Returns:
            Список результатов поиска
        """
        results = []

        # 1. Извлекаем сущности из запроса
        entities = None
        if use_entities:
            entities = self.entity_extractor.extract(query)
            logger.info(f"Извлечены сущности: {entities.to_dict()}")

        # 2. Ищем узлы по сущностям
        candidate_nodes = self._find_candidate_nodes(entities, filters)

        # 3. Ранжируем узлы
        scored_nodes = self._score_nodes(candidate_nodes, entities, query)

        # 4. Расширяем на связанные узлы если нужно
        if expand_graph and len(scored_nodes) > 0:
            expanded_nodes = self._expand_to_related(
                scored_nodes[:5], entities, query)
            scored_nodes.extend(expanded_nodes)

        # 5. Сортируем и берем top_k
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        # 6. Формируем результаты
        for node_id, score in scored_nodes[:top_k]:
            result = self._create_search_result(node_id, score)
            if result:
                results.append(result)

        return results

    def _find_candidate_nodes(self,
                              entities: Optional[ExtractedEntities],
                              filters: Optional[Dict[str, Any]]) -> List[str]:
        """Поиск узлов-кандидатов по сущностям и фильтрам"""
        candidates = set()

        if entities:
            # Поиск по домену
            if entities.domain:
                domain_nodes = self.graph_builder.find_nodes_by_domain(
                    entities.domain.value)
                candidates.update(domain_nodes)

            # Поиск по объектам
            for obj in entities.object_attributes:
                nodes = self.graph_builder.find_nodes_by_entities(
                    "object", obj)
                candidates.update(nodes)

            # Поиск по участникам
            for actor in entities.actors:
                nodes = self.graph_builder.find_nodes_by_entities(
                    "actors", actor)
                candidates.update(nodes)

            # Поиск по идентификаторам
            for identifier in entities.identifiers:
                nodes = self.graph_builder.find_nodes_by_entities(
                    "identifiers", identifier)
                candidates.update(nodes)

        # Если кандидатов мало, добавляем все документы
        if len(candidates) < 5:
            for node_id in self.graph_builder.graph.nodes():
                node_data = self.graph_builder.graph.nodes[node_id]
                if node_data.get("type") in ["document", "section"]:
                    candidates.add(node_id)

        # Применяем фильтры
        if filters:
            candidates = self._apply_filters(candidates, filters)

        return list(candidates)

    def _apply_filters(self,
                       candidates: set,
                       filters: Dict[str, Any]) -> set:
        """Применение фильтров к кандидатам"""
        filtered = set()

        for node_id in candidates:
            node_data = self.graph_builder.graph.nodes[node_id]

            # Фильтр по типу документа
            if "doc_type" in filters:
                if node_data.get("doc_type") != filters["doc_type"]:
                    continue

            # Фильтр по дате
            if "date_range" in filters:
                node_date = node_data.get("date")
                if not node_date:
                    continue
                # Здесь нужна логика сравнения дат
                # TODO: реализовать сравнение дат

            # Фильтр по пути
            if "path_prefix" in filters:
                node_path = node_data.get("path", "")
                if not node_path.startswith(filters["path_prefix"]):
                    continue

            filtered.add(node_id)

        return filtered

    def _score_nodes(self,
                     candidates: List[str],
                     entities: Optional[ExtractedEntities],
                     query: str) -> List[Tuple[str, float]]:
        """Ранжирование узлов-кандидатов"""
        scored = []

        for node_id in candidates:
            score = self._calculate_node_score(node_id, entities, query)
            if score > 0:
                scored.append((node_id, score))

        return scored

    def _calculate_node_score(self,
                              node_id: str,
                              entities: Optional[ExtractedEntities],
                              query: str) -> float:
        """Вычисление релевантности узла"""
        if not self.graph_builder.graph.has_node(node_id):
            return 0.0

        node_data = self.graph_builder.graph.nodes[node_id]
        score = 0.0

        # 1. Базовая важность узла
        importance = self.graph_builder.get_node_importance(node_id)
        score += importance * 0.2

        # 2. PageRank если есть
        if node_id in self.pagerank_scores:
            score += self.pagerank_scores[node_id] * 0.1

        # 3. Совпадение по сущностям
        if entities and "entities" in node_data:
            node_entities = node_data["entities"]
            if isinstance(node_entities, str):
                node_entities = json.loads(node_entities)

            entity_score = self._calculate_entity_match_score(
                entities, node_entities)
            score += entity_score * 0.5

        # 4. Текстовое совпадение (простое)
        if "content" in node_data:
            content = node_data["content"].lower()
            query_lower = query.lower()
            query_words = query_lower.split()

            # Считаем совпадения слов
            matches = sum(1 for word in query_words if word in content)
            text_score = matches / len(query_words) if query_words else 0
            score += text_score * 0.2

        return score

    def _calculate_entity_match_score(self,
                                      query_entities: ExtractedEntities,
                                      node_entities: Dict[str, Any]) -> float:
        """Вычисление совпадения сущностей"""
        score = 0.0
        matches = 0
        total = 0

        # Сравниваем домены
        if query_entities.domain and node_entities.get("domain"):
            total += 1
            if query_entities.domain.value == node_entities["domain"]:
                matches += 1
                score += 0.3

        # Сравниваем объекты
        if query_entities.object_attributes and node_entities.get("object_attributes"):
            total += 1
            common = set(query_entities.object_attributes) & set(
                node_entities["object_attributes"])
            if common:
                matches += len(common) / len(query_entities.object_attributes)
                score += 0.2

        # Сравниваем участников
        if query_entities.actors and node_entities.get("actors"):
            total += 1
            common = set(query_entities.actors) & set(node_entities["actors"])
            if common:
                matches += len(common) / len(query_entities.actors)
                score += 0.2

        # Сравниваем идентификаторы
        if query_entities.identifiers and node_entities.get("identifiers"):
            total += 1
            common = set(query_entities.identifiers) & set(
                node_entities["identifiers"])
            if common:
                matches += 1  # Идентификаторы важнее
                score += 0.3

        # Нормализуем
        if total > 0:
            score = score * (matches / total)

        return min(score, 1.0)

    def _expand_to_related(self,
                           top_nodes: List[Tuple[str, float]],
                           entities: Optional[ExtractedEntities],
                           query: str,
                           max_distance: int = 2) -> List[Tuple[str, float]]:
        """Расширение поиска на связанные узлы"""
        expanded = []
        seen = set([node_id for node_id, _ in top_nodes])

        for node_id, base_score in top_nodes:
            related = self.graph_builder.get_related_nodes(
                node_id, max_distance)

            for related_id, distance in related:
                if related_id not in seen:
                    # Уменьшаем score в зависимости от расстояния
                    distance_penalty = 1.0 / (distance + 1)
                    related_score = self._calculate_node_score(
                        related_id, entities, query)
                    final_score = related_score * distance_penalty * \
                        0.7  # 0.7 - штраф за непрямое совпадение

                    if final_score > 0.1:  # Порог отсечения
                        expanded.append((related_id, final_score))
                        seen.add(related_id)

        return expanded

    def _create_search_result(self, node_id: str, score: float) -> Optional[GraphSearchResult]:
        """Создание результата поиска из узла графа"""
        if not self.graph_builder.graph.has_node(node_id):
            return None

        node_data = self.graph_builder.graph.nodes[node_id]

        # Получаем связанные узлы
        related = self.graph_builder.get_related_nodes(node_id, max_distance=1)
        related_ids = [rid for rid, _ in related[:5]]  # Топ-5 связанных

        # Формируем результат
        result = GraphSearchResult(
            node_id=node_id,
            node_type=node_data.get("type", "unknown"),
            score=score,
            path=node_data.get("path"),
            title=node_data.get("title") or node_data.get("name"),
            content=node_data.get("content"),
            entities=json.loads(node_data["entities"]) if "entities" in node_data and isinstance(
                node_data["entities"], str) else node_data.get("entities"),
            importance=self.graph_builder.get_node_importance(node_id),
            related_nodes=related_ids,
            metadata={
                k: v for k, v in node_data.items()
                if k not in ["content", "entities", "type", "path", "title", "name"]
            }
        )

        return result

    def get_document_hierarchy(self, doc_id: str) -> Dict[str, Any]:
        """
        Получение иерархии документа

        Args:
            doc_id: ID документа

        Returns:
            Иерархическая структура документа
        """
        if not self.graph_builder.graph.has_node(doc_id):
            return {}

        hierarchy = {
            "id": doc_id,
            "type": self.graph_builder.graph.nodes[doc_id].get("type"),
            "title": self.graph_builder.graph.nodes[doc_id].get("title"),
            "children": []
        }

        # Получаем дочерние узлы
        for child_id in self.graph_builder.graph.successors(doc_id):
            child_data = self.get_document_hierarchy(child_id)
            hierarchy["children"].append(child_data)

        return hierarchy

    def find_similar_documents(self, doc_id: str, top_k: int = 5) -> List[GraphSearchResult]:
        """
        Поиск похожих документов

        Args:
            doc_id: ID документа
            top_k: Количество результатов

        Returns:
            Список похожих документов
        """
        if not self.graph_builder.graph.has_node(doc_id):
            return []

        node_data = self.graph_builder.graph.nodes[doc_id]

        # Извлекаем сущности документа
        doc_entities = node_data.get("entities")
        if not doc_entities:
            return []

        if isinstance(doc_entities, str):
            doc_entities = json.loads(doc_entities)

        # Создаем псевдо-запрос из сущностей документа
        pseudo_query = []
        if doc_entities.get("object_attributes"):
            pseudo_query.extend(doc_entities["object_attributes"])
        if doc_entities.get("actors"):
            pseudo_query.extend(doc_entities["actors"])

        query = " ".join(pseudo_query)

        # Ищем похожие, исключая сам документ
        results = self.search(query, top_k=top_k+1, use_entities=False)
        return [r for r in results if r.node_id != doc_id][:top_k]


# Функция для тестирования
def test_graph_search():
    """Тестирование поиска по графу"""
    searcher = GraphSearcher()

    test_queries = [
        "Договор поставки цемента М500",
        "Документы по логистике",
        "Железнодорожные перевозки",
    ]

    for query in test_queries:
        print(f"\nЗапрос: {query}")
        print("-" * 50)
        results = searcher.search(query, top_k=5)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title or result.node_id}")
            print(f"   Тип: {result.node_type}, Score: {result.score:.3f}")
            print(f"   Путь: {result.path}")
            if result.entities:
                print(f"   Сущности: {result.entities}")
            print()


if __name__ == "__main__":
    test_graph_search()
