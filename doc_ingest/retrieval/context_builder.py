# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Tuple, Any, Dict
from pydantic import BaseModel
from collections import defaultdict
import re

from config import ContextConfig
from retrieval.reranker import RerankedDoc


class ContextItem(BaseModel):
    doc_id: Optional[str]
    doc_path: Optional[str]
    category: Optional[str]
    text: str
    page_span: Optional[Tuple[Optional[int], Optional[int]]] = None
    heading_path: Optional[List[str]] = None
    chunk_index: Optional[int] = None
    score: Optional[float] = None
    hierarchy_level: Optional[int] = None
    section_id: Optional[str] = None


class ContextPack(BaseModel):
    items: List[ContextItem]
    prompt: str                  # склеенный контекст (для RAG)
    citations: List[str]         # список цитат в порядке включения
    hierarchy_summary: str       # краткое описание структуры
    coverage_stats: Dict[str, Any]  # статистика покрытия


class HierarchicalContextBuilder:
    """
    Улучшенный контекст-билдер с учетом иерархии документов:
    - Группировка по разделам и подразделам
    - Сохранение структурной связности
    - Умное усечение с сохранением контекста
    - Анализ покрытия тематики
    """

    def __init__(self, cfg: ContextConfig, tokenizer=None):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def _toklen(self, text: str) -> int:
        """Подсчет токенов в тексте"""
        if self.tokenizer is None:
            return max(1, int(len(text) / 4))
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _extract_section_id(self, heading_path: Optional[List[str]]) -> str:
        """Извлечение идентификатора раздела"""
        if not heading_path:
            return "root"

        # Ищем номер раздела (например, "1.2.3")
        for heading in heading_path:
            match = re.search(r'^(\d+(?:\.\d+)*)', heading.strip())
            if match:
                return match.group(1)

        # Если нет номера, используем первый заголовок
        return heading_path[0][:20].replace(" ", "_")

    def _group_by_sections(self, docs: List[RerankedDoc]) -> Dict[str, List[Tuple[RerankedDoc, Any]]]:
        """Группировка документов по разделам"""
        sections = defaultdict(list)

        for doc in docs:
            for hit in doc.hits:
                section_id = self._extract_section_id(hit.heading_path)
                sections[section_id].append((doc, hit))

        return sections

    def _compute_hierarchy_level(self, heading_path: Optional[List[str]]) -> int:
        """Вычисление уровня иерархии"""
        if not heading_path:
            return 0

        # Считаем глубину по количеству уровней
        max_level = 0
        for heading in heading_path:
            # Ищем номер раздела типа "1.2.3"
            match = re.search(r'^(\d+(?:\.\d+)*)', heading.strip())
            if match:
                level = len(match.group(1).split('.'))
                max_level = max(max_level, level)

        return max_level

    def _smart_truncate(self, text: str, max_tokens: int, preserve_sentences: bool = True) -> str:
        """Умное усечение текста с сохранением структуры"""
        if self._toklen(text) <= max_tokens:
            return text

        if preserve_sentences:
            # Разбиваем на предложения
            sentences = re.split(r'[.!?]+', text)
            truncated = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                test_text = truncated + sentence + ". "
                if self._toklen(test_text) <= max_tokens:
                    truncated = test_text
                else:
                    break

            if truncated:
                return truncated.strip()

        # Fallback: усечение по символам
        return text[:max_tokens * 4]

    def _build_citation(self, doc: RerankedDoc, hit: Any) -> str:
        """Построение цитаты для документа"""
        cite_parts = []

        if doc.category:
            cite_parts.append(doc.category)
        if doc.doc_path:
            cite_parts.append(doc.doc_path)

        if self.cfg.cite_with_pages and (hit.page_from is not None or hit.page_to is not None):
            if hit.page_from == hit.page_to or hit.page_to is None:
                cite_parts.append(f"p. {hit.page_from}")
            else:
                cite_parts.append(f"pp. {hit.page_from}–{hit.page_to}")

        return " | ".join(cite_parts) if cite_parts else ""

    def _analyze_coverage(self, items: List[ContextItem]) -> Dict[str, Any]:
        """Анализ покрытия тематики"""
        categories = defaultdict(int)
        sections = defaultdict(int)
        hierarchy_levels = defaultdict(int)

        for item in items:
            if item.category:
                categories[item.category] += 1
            if item.section_id:
                sections[item.section_id] += 1
            if item.hierarchy_level is not None:
                hierarchy_levels[item.hierarchy_level] += 1

        return {
            "total_items": len(items),
            "categories": dict(categories),
            "sections": dict(sections),
            "hierarchy_levels": dict(hierarchy_levels),
            "category_coverage": len(categories),
            "section_coverage": len(sections)
        }

    def _build_hierarchy_summary(self, items: List[ContextItem]) -> str:
        """Построение краткого описания структуры"""
        if not self.cfg.preserve_hierarchy:
            return ""

        # Группируем по категориям
        by_category = defaultdict(list)
        for item in items:
            if item.category:
                by_category[item.category].append(item)

        summary_parts = []
        for category, cat_items in by_category.items():
            # Находим основные разделы
            sections = set()
            for item in cat_items:
                if item.section_id and item.section_id != "root":
                    sections.add(item.section_id)

            if sections:
                summary_parts.append(
                    f"{category}: разделы {', '.join(sorted(sections))}")
            else:
                summary_parts.append(f"{category}: общая информация")

        return "; ".join(summary_parts)

    def build(self, docs: List[RerankedDoc]) -> ContextPack:
        """Основной метод построения контекста"""
        if not docs:
            return ContextPack(
                items=[], prompt="", citations=[],
                hierarchy_summary="", coverage_stats={}
            )

        budget = self.cfg.token_budget
        items: List[ContextItem] = []
        citations: List[str] = []

        # Группируем по разделам если включено
        if self.cfg.group_by_sections:
            sections = self._group_by_sections(docs)
            # Сортируем разделы по важности (количество документов)
            sorted_sections = sorted(sections.items(),
                                     key=lambda x: len(x[1]), reverse=True)
        else:
            # Простая обработка по документам
            sections = {"all": [(doc, hit)
                                for doc in docs for hit in doc.hits]}
            sorted_sections = [("all", sections["all"])]

        # Обрабатываем разделы
        for section_id, section_items in sorted_sections:
            if budget <= 0:
                break

            # Сортируем элементы раздела по скору
            section_items.sort(key=lambda x: x[1].score or 0, reverse=True)

            # Резервируем бюджет для других разделов
            section_budget = int(budget * 0.8)

            for doc, hit in section_items:
                if budget <= 0:
                    break

                # Строим цитату
                citation = self._build_citation(doc, hit)

                # Формируем текст с заголовками
                text = hit.text
                if self.cfg.include_headings and hit.heading_path:
                    text = f"{' > '.join(hit.heading_path)}\n{text}"

                # Проверяем размер
                need = self._toklen(text)

                if need > budget:
                    # Умное усечение
                    if self.cfg.enable_smart_truncation:
                        text = self._smart_truncate(text, budget)
                        need = self._toklen(text)
                    else:
                        continue

                if need <= budget:
                    # Вычисляем уровень иерархии
                    hierarchy_level = self._compute_hierarchy_level(
                        hit.heading_path)

                    items.append(ContextItem(
                        doc_id=doc.doc_id,
                        doc_path=doc.doc_path,
                        category=doc.category,
                        text=text,
                        page_span=(hit.page_from, hit.page_to),
                        heading_path=hit.heading_path or None,
                        chunk_index=hit.chunk_index,
                        score=hit.score,
                        hierarchy_level=hierarchy_level,
                        section_id=section_id
                    ))

                    budget -= need
                    if citation:
                        citations.append(citation)

                    # Проверяем минимальное покрытие раздела
                    if (self.cfg.min_section_coverage > 0 and
                            len([i for i in items if i.section_id == section_id]) >= 2):
                        break

        # Анализируем покрытие
        coverage_stats = self._analyze_coverage(items)
        hierarchy_summary = self._build_hierarchy_summary(items)

        # Строим финальный промпт
        if self.cfg.preserve_hierarchy:
            # Группируем по разделам в промпте
            by_section = defaultdict(list)
            for item in items:
                by_section[item.section_id or "general"].append(item)

            prompt_parts = []
            for section_id in sorted(by_section.keys()):
                section_items = by_section[section_id]
                if section_id != "general":
                    prompt_parts.append(f"=== Раздел: {section_id} ===")

                section_texts = [item.text for item in section_items]
                prompt_parts.append(
                    self.cfg.join_separator.join(section_texts))

            prompt = "\n\n".join(prompt_parts)
        else:
            # Простое объединение
            prompt = self.cfg.join_separator.join(
                [item.text for item in items])

        return ContextPack(
            items=items,
            prompt=prompt,
            citations=citations,
            hierarchy_summary=hierarchy_summary,
            coverage_stats=coverage_stats
        )

    def get_quality_metrics(self, context_pack: ContextPack) -> Dict[str, float]:
        """Вычисление метрик качества контекста"""
        if not context_pack.items:
            return {}

        # Разнообразие источников
        source_diversity = len(set(item.doc_id for item in context_pack.items))

        # Покрытие иерархии
        hierarchy_coverage = len(set(
            item.hierarchy_level for item in context_pack.items if item.hierarchy_level is not None))

        # Баланс категорий
        categories = [
            item.category for item in context_pack.items if item.category]
        category_balance = len(set(categories)) / max(len(categories), 1)

        # Средний скор релевантности
        scores = [
            item.score for item in context_pack.items if item.score is not None]
        avg_relevance = sum(scores) / len(scores) if scores else 0.0

        return {
            "source_diversity": source_diversity,
            "hierarchy_coverage": hierarchy_coverage,
            "category_balance": category_balance,
            "avg_relevance": avg_relevance,
            "total_items": len(context_pack.items)
        }


# Обратная совместимость
ContextBuilder = HierarchicalContextBuilder
