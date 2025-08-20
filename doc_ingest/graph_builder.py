# graph_builder.py
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from models import DocElement, Chunk
from pathlib import Path
import json
import re
from collections import defaultdict


class HierarchicalGraphBuilder:
    """
    Улучшенный графовый анализатор для учета иерархии документов:
    - Анализ структурной связности
    - Вычисление важности разделов
    - Построение семантических связей
    - Оптимизация для поиска
    - Интеграция с системой извлечения сущностей
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.section_importance = {}
        self.hierarchy_levels = {}
        self.semantic_connections = {}
        self.entity_index = {}  # Индекс сущностей для быстрого поиска
        self.domain_index = defaultdict(list)  # Индекс по доменам

    def _sanitize(self, value: Any) -> Any:
        """Санитизация значений для GraphML"""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    def _clean_attrs(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Очистка атрибутов"""
        out = {}
        for k, v in d.items():
            sv = self._sanitize(v)
            if sv is None:
                continue
            out[k] = sv
        return out

    def _extract_section_number(self, heading: str) -> Optional[str]:
        """Извлечение номера раздела"""
        if not heading:
            return None

        # Ищем номера разделов типа "1.2.3" или "Раздел 1"
        patterns = [
            r'^(\d+(?:\.\d+)*)',  # 1.2.3
            r'^(раздел|глава|приложение)\s+(\d+)',  # Раздел 1
            r'^(\d+)[\)\.]',  # 1) или 1.
        ]

        for pattern in patterns:
            match = re.search(pattern, heading.strip(), re.IGNORECASE)
            if match:
                return match.group(1) if pattern != patterns[1] else match.group(2)

        return None

    def _compute_hierarchy_level(self, heading_path: List[str]) -> int:
        """Вычисление уровня иерархии"""
        if not heading_path:
            return 0

        max_level = 0
        for heading in heading_path:
            section_num = self._extract_section_number(heading)
            if section_num:
                level = len(section_num.split('.'))
                max_level = max(max_level, level)

        return max_level

    def _compute_section_importance(self, elements: List[DocElement]) -> Dict[str, float]:
        """Вычисление важности разделов"""
        section_stats = defaultdict(lambda: {
            'count': 0,
            'total_length': 0,
            'has_tables': 0,
            'has_formulas': 0,
            'has_images': 0
        })

        for element in elements:
            section_id = self._extract_section_number(element.text or "")
            if not section_id:
                section_id = "general"

            stats = section_stats[section_id]
            stats['count'] += 1
            stats['total_length'] += len(element.text or "")

            if element.element_type == "table":
                stats['has_tables'] += 1
            elif element.element_type == "formula":
                stats['has_formulas'] += 1
            elif element.element_type == "image":
                stats['has_images'] += 1

        # Вычисляем важность
        importance = {}
        for section_id, stats in section_stats.items():
            # Важность = количество элементов + длина текста + специальные элементы
            importance[section_id] = (
                stats['count'] * 0.3 +
                min(stats['total_length'] / 1000, 1.0) * 0.4 +
                (stats['has_tables'] + stats['has_formulas'] +
                 stats['has_images']) * 0.3
            )

        return importance

    def _build_semantic_connections(self, elements: List[DocElement]) -> Dict[str, List[str]]:
        """Построение семантических связей между разделами"""
        connections = defaultdict(list)

        # Анализируем ссылки и перекрестные упоминания
        for element in elements:
            if not element.text:
                continue

            # Ищем ссылки на другие разделы
            section_refs = re.findall(
                r'(?:раздел|глава|пункт)\s+(\d+(?:\.\d+)*)', element.text, re.IGNORECASE)
            current_section = self._extract_section_number(element.text)

            if current_section and section_refs:
                for ref in section_refs:
                    if ref != current_section:
                        connections[current_section].append(ref)

        return dict(connections)

    def build_graph(self, elements_by_doc: Dict[str, List[DocElement]]) -> nx.DiGraph:
        """
        Построение графа с учетом иерархии:
        - Структурная иерархия папок и документов
        - Иерархия разделов внутри документов
        - Семантические связи между разделами
        """
        self.graph = nx.DiGraph()

        for doc_key, elems in elements_by_doc.items():
            cat_raw, doc_path = doc_key.split("::", 1)
            cat_parts = [p for p in Path(cat_raw).parts if p not in (".",)]

            # Строим иерархию папок
            parent_id = None
            accum = []
            for i, part in enumerate(cat_parts):
                accum.append(part)
                node_id = f"folder::{'/'.join(accum)}"

                if not self.graph.has_node(node_id):
                    self.graph.add_node(node_id, **self._clean_attrs({
                        "type": "folder",
                        "name": part,
                        "path": "/".join(accum),
                        "level": i,
                        # Папки верхнего уровня важнее
                        "importance": 1.0 / (i + 1)
                    }))

                if parent_id:
                    if not self.graph.has_edge(parent_id, node_id):
                        self.graph.add_edge(parent_id, node_id, **self._clean_attrs({
                            "rel": "refines",
                            "weight": 1.0
                        }))
                parent_id = node_id

            # Документ
            doc_id = f"document::{doc_path}"
            if not self.graph.has_node(doc_id):
                self.graph.add_node(doc_id, **self._clean_attrs({
                    "type": "document",
                    "path": doc_path,
                    "category": cat_raw,
                    "element_count": len(elems)
                }))

            if parent_id:
                if not self.graph.has_edge(parent_id, doc_id):
                    self.graph.add_edge(parent_id, doc_id, **self._clean_attrs({
                        "rel": "contains",
                        "weight": 1.0
                    }))

            # Анализируем элементы документа
            section_importance = self._compute_section_importance(elems)
            semantic_connections = self._build_semantic_connections(elems)

            # Сохраняем метаданные
            self.section_importance[doc_id] = section_importance
            self.semantic_connections[doc_id] = semantic_connections

            # Строим иерархию разделов
            sections_by_level = defaultdict(list)
            for element in elems:
                if element.headings_path:
                    level = self._compute_hierarchy_level(
                        element.headings_path)
                    sections_by_level[level].append(element)

            # Добавляем разделы в граф
            for level, level_elements in sections_by_level.items():
                for element in level_elements:
                    section_id = self._extract_section_number(
                        element.text or "")
                    if not section_id:
                        section_id = f"section_{element.id[:8]}"

                    section_node_id = f"section::{doc_id}::{section_id}"

                    if not self.graph.has_node(section_node_id):
                        importance = section_importance.get(section_id, 0.5)
                        self.graph.add_node(section_node_id, **self._clean_attrs({
                            "type": "section",
                            "section_id": section_id,
                            "level": level,
                            "importance": importance,
                            "heading": " > ".join(element.headings_path or [])[:100]
                        }))

                        # Связь с документом
                        self.graph.add_edge(doc_id, section_node_id, **self._clean_attrs({
                            "rel": "has_section",
                            "weight": importance
                        }))

                    # Элемент
                    if not self.graph.has_node(element.id):
                        self.graph.add_node(element.id, **self._clean_attrs({
                            "type": element.element_type,
                            "page": element.page,
                            "order": element.order,
                            "text_length": len(element.text or ""),
                            "has_media": bool(element.media_path)
                        }))

                        # Связь с разделом
                        self.graph.add_edge(section_node_id, element.id, **self._clean_attrs({
                            "rel": "contains_element",
                            "weight": 1.0
                        }))

            # Добавляем семантические связи между разделами
            for section_id, connections in semantic_connections.items():
                source_node = f"section::{doc_id}::{section_id}"
                if self.graph.has_node(source_node):
                    for target_section in connections:
                        target_node = f"section::{doc_id}::{target_section}"
                        if self.graph.has_node(target_node):
                            self.graph.add_edge(source_node, target_node, **self._clean_attrs({
                                "rel": "references",
                                "weight": 0.8,
                                "semantic": True
                            }))

        return self.graph

    def analyze_hierarchy(self) -> Dict[str, Any]:
        """Анализ иерархии графа"""
        if not self.graph.nodes():
            return {}

        analysis = {
            "total_nodes": len(self.graph.nodes()),
            "total_edges": len(self.graph.edges()),
            "node_types": defaultdict(int),
            "hierarchy_depth": 0,
            "section_importance": {},
            "semantic_connections": {}
        }

        # Анализируем типы узлов
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            analysis["node_types"][node_type] += 1

            # Максимальная глубина иерархии
            if node_type == "section":
                level = attrs.get("level", 0)
                analysis["hierarchy_depth"] = max(
                    analysis["hierarchy_depth"], level)

        # Анализируем важность разделов
        for doc_id, importance in self.section_importance.items():
            analysis["section_importance"][doc_id] = {
                "sections": len(importance),
                "avg_importance": sum(importance.values()) / len(importance) if importance else 0,
                "top_sections": sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            }

        # Анализируем семантические связи
        for doc_id, connections in self.semantic_connections.items():
            analysis["semantic_connections"][doc_id] = {
                "total_connections": sum(len(conns) for conns in connections.values()),
                "connected_sections": len(connections),
                "avg_connections_per_section": sum(len(conns) for conns in connections.values()) / len(connections) if connections else 0
            }

        return analysis

    def get_relevant_sections(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Поиск релевантных разделов по запросу"""
        if not self.graph.nodes():
            return []

        # Простой поиск по ключевым словам
        query_tokens = set(re.findall(r'\w+', query.lower()))
        section_scores = []

        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "section":
                section_text = attrs.get("heading", "")
                section_tokens = set(re.findall(r'\w+', section_text.lower()))

                # Вычисляем пересечение токенов
                overlap = len(query_tokens & section_tokens)
                if overlap > 0:
                    # Скор = пересечение токенов * важность раздела
                    importance = attrs.get("importance", 0.5)
                    score = overlap * importance
                    section_scores.append((node, score))

        # Сортируем по скору
        section_scores.sort(key=lambda x: x[1], reverse=True)
        return section_scores[:top_k]

    def get_structural_path(self, from_node: str, to_node: str) -> List[str]:
        """Поиск структурного пути между узлами"""
        try:
            path = nx.shortest_path(self.graph, from_node, to_node)
            return path
        except nx.NetworkXNoPath:
            return []

    def save_graph(self, name: str = "corpus_graph"):
        """Сохранение графа в различных форматах"""
        if not self.graph.nodes():
            print("Граф пуст, нечего сохранять")
            return

        # Санитизация всех узлов/рёбер для GraphML
        for n, attrs in list(self.graph.nodes(data=True)):
            self.graph.nodes[n].clear()
            self.graph.nodes[n].update(self._clean_attrs(attrs))

        for u, v, attrs in list(self.graph.edges(data=True)):
            self.graph.edges[u, v].clear()
            self.graph.edges[u, v].update(self._clean_attrs(attrs))

        # Создаем директорию
        Path("outputs/graphs").mkdir(parents=True, exist_ok=True)

        # Сохраняем в различных форматах
        nx.write_graphml(self.graph, f"outputs/graphs/{name}.graphml")
        nx.write_gexf(self.graph, f"outputs/graphs/{name}.gexf")

        # JSON с дополнительной информацией
        data = nx.readwrite.json_graph.node_link_data(self.graph)
        graph_data = {
            "graph": data,
            "analysis": self.analyze_hierarchy(),
            "section_importance": self.section_importance,
            "semantic_connections": self.semantic_connections
        }

        with open(f"outputs/graphs/{name}.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        print(f"Граф сохранен в outputs/graphs/{name}.*")

    def add_entities_to_graph(self, node_id: str, entities: Dict[str, Any]):
        """
        Добавление извлеченных сущностей к узлу графа

        Args:
            node_id: ID узла в графе
            entities: Словарь с извлеченными сущностями
        """
        if not self.graph.has_node(node_id):
            return

        # Добавляем сущности как атрибуты узла
        self.graph.nodes[node_id]['entities'] = entities

        # Индексируем для быстрого поиска
        if entities.get('domain'):
            self.domain_index[entities['domain']].append(node_id)

        # Индексируем по ключевым сущностям
        for entity_type in ['object', 'actors', 'identifiers']:
            if entities.get(entity_type):
                if entity_type not in self.entity_index:
                    self.entity_index[entity_type] = defaultdict(list)

                items = entities[entity_type]
                if isinstance(items, list):
                    for item in items:
                        self.entity_index[entity_type][item].append(node_id)
                else:
                    self.entity_index[entity_type][items].append(node_id)

    def find_nodes_by_entities(self, entity_type: str, entity_value: str) -> List[str]:
        """
        Поиск узлов по сущностям

        Args:
            entity_type: Тип сущности (object, actors, identifiers и т.д.)
            entity_value: Значение сущности

        Returns:
            Список ID узлов, содержащих данную сущность
        """
        if entity_type in self.entity_index:
            return self.entity_index[entity_type].get(entity_value, [])
        return []

    def find_nodes_by_domain(self, domain: str) -> List[str]:
        """
        Поиск узлов по домену

        Args:
            domain: Название домена

        Returns:
            Список ID узлов данного домена
        """
        return self.domain_index.get(domain, [])

    def get_node_importance(self, node_id: str) -> float:
        """
        Получение важности узла на основе его положения в графе

        Args:
            node_id: ID узла

        Returns:
            Коэффициент важности (0-1)
        """
        if not self.graph.has_node(node_id):
            return 0.0

        # Базовая важность из атрибутов
        base_importance = self.graph.nodes[node_id].get('importance', 0.5)

        # Учитываем PageRank если граф достаточно большой
        if len(self.graph) > 10:
            try:
                pagerank = nx.pagerank(self.graph, alpha=0.85)
                pr_importance = pagerank.get(node_id, 0.0)
                # Комбинируем базовую важность и PageRank
                return 0.7 * base_importance + 0.3 * pr_importance
            except:
                pass

        return base_importance

    def get_related_nodes(self, node_id: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """
        Получение связанных узлов в пределах заданного расстояния

        Args:
            node_id: ID начального узла
            max_distance: Максимальное расстояние в графе

        Returns:
            Список кортежей (node_id, distance)
        """
        if not self.graph.has_node(node_id):
            return []

        related = []
        distances = nx.single_source_shortest_path_length(
            self.graph, node_id, cutoff=max_distance
        )

        for target, distance in distances.items():
            if target != node_id:
                related.append((target, distance))

        # Сортируем по расстоянию
        related.sort(key=lambda x: x[1])
        return related

    def load_graph(self, name="corpus_graph") -> bool:
        """
        Загрузка сохраненного графа

        Args:
            name: Имя файла графа

        Returns:
            True если граф успешно загружен
        """
        graph_dir = Path("outputs/graphs")
        graphml_path = graph_dir / f"{name}.graphml"

        if graphml_path.exists():
            try:
                self.graph = nx.read_graphml(str(graphml_path))

                # Восстанавливаем индексы
                self.entity_index = {}
                self.domain_index = defaultdict(list)

                for node_id in self.graph.nodes():
                    node_data = self.graph.nodes[node_id]
                    if 'entities' in node_data:
                        # Десериализуем JSON если нужно
                        entities = node_data['entities']
                        if isinstance(entities, str):
                            entities = json.loads(entities)
                        self.add_entities_to_graph(node_id, entities)

                print(f"Граф загружен из {graphml_path}")
                return True
            except Exception as e:
                print(f"Ошибка загрузки графа: {e}")
                return False
        return False


# Обратная совместимость
def build_graph(elements_by_doc: Dict[str, List[DocElement]]) -> nx.DiGraph:
    """Функция для обратной совместимости"""
    builder = HierarchicalGraphBuilder()
    return builder.build_graph(elements_by_doc)


def save_graph(G: nx.DiGraph, name="corpus_graph"):
    """Функция для обратной совместимости"""
    # Создаем временный builder для сохранения
    builder = HierarchicalGraphBuilder()
    builder.graph = G
    builder.save_graph(name)
