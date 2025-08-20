"""
Система извлечения сущностей из пользовательских запросов
Комбинирует правила, NER и LLM для максимальной точности
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
import logging
from enum import Enum

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Domain(Enum):
    """Домены/области бизнеса"""
    LEGAL = "Юристы"
    LOGISTICS = "Логистика"
    PROCUREMENT = "Закупки"
    COMMERCE = "Коммерция"
    PRODUCTION = "Производство"
    FINANCE = "Финансы"
    PR = "PR"
    HR = "HR"
    UNKNOWN = "Неизвестно"


class ObjectType(Enum):
    """Типы объектов"""
    PRODUCT = "товар"
    SERVICE = "услуга"
    DOCUMENT = "документ"
    PROCESS = "процесс"
    UNKNOWN = "неизвестно"


class Action(Enum):
    """Типы действий"""
    SEARCH = "поиск"
    COMPARE = "сравнение"
    CHECK = "проверка"
    GENERATE = "генерация"
    SUMMARIZE = "суммаризация"
    CALCULATE = "расчет"
    UNKNOWN = "неизвестно"


@dataclass
class ExtractedEntities:
    """Извлеченные сущности из запроса"""

    # Основные поля
    intent: Optional[str] = None
    domain: Optional[Domain] = None
    object_type: Optional[ObjectType] = None
    action: Optional[Action] = None

    # Атрибуты и идентификаторы
    object_attributes: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)

    # Участники и локации
    actors: List[str] = field(default_factory=list)
    # {"origin": ..., "destination": ...}
    location: Optional[Dict[str, str]] = None

    # Временные параметры
    # {"from": ..., "to": ..., "deadline": ...}
    time: Optional[Dict[str, Any]] = None

    # Количественные параметры
    quantity: Optional[Dict[str, Any]] = None  # {"value": ..., "unit": ...}
    # {"amount": ..., "currency": ..., "vat": ...}
    price: Optional[Dict[str, Any]] = None

    # Метрики и ограничения
    metrics: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)

    # Метаинформация
    output_format: Optional[str] = None
    locale: str = "ru-RU"
    priority: str = "normal"  # low/normal/high
    authority: List[str] = field(default_factory=list)
    context_refs: List[str] = field(default_factory=list)
    sensitivity: str = "normal"  # normal/confidential/pii
    needs_grounding: bool = False

    # Уверенность в извлечении
    confidences: Dict[str, float] = field(default_factory=dict)

    # Доменно-специфичные поля
    legal_info: Optional[Dict[str, Any]] = None
    logistics_info: Optional[Dict[str, Any]] = None
    procurement_info: Optional[Dict[str, Any]] = None
    manufacturing_info: Optional[Dict[str, Any]] = None
    finance_info: Optional[Dict[str, Any]] = None
    hr_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для JSON"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and value != [] and value != {}:
                if isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
        return result


class EntityExtractor:
    """Главный класс для извлечения сущностей"""

    def __init__(self,
                 use_llm: bool = True,
                 llm_model_path: Optional[str] = None,
                 dictionaries_path: Optional[str] = None):
        """
        Args:
            use_llm: Использовать ли LLM для извлечения
            llm_model_path: Путь к локальной LLM модели
            dictionaries_path: Путь к справочникам и онтологиям
        """
        self.use_llm = use_llm
        self.llm_model_path = llm_model_path
        self.dictionaries_path = dictionaries_path or "doc_ingest/dictionaries"

        # Загрузка справочников
        self._load_dictionaries()

        # Инициализация LLM если нужно
        if self.use_llm:
            self._init_llm()

    def _load_dictionaries(self):
        """Загрузка справочников и онтологий"""
        self.dictionaries = {}
        dict_path = Path(self.dictionaries_path)

        if dict_path.exists():
            for file in dict_path.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    self.dictionaries[file.stem] = json.load(f)

        # Дефолтные справочники если файлов нет
        if not self.dictionaries:
            self.dictionaries = self._get_default_dictionaries()

    def _get_default_dictionaries(self) -> Dict[str, Any]:
        """Дефолтные справочники"""
        return {
            "products": {
                "цемент": ["М500", "М400", "М300", "ПЦ", "ШПЦ", "БЦ", "ГОСТ 31108-2020"],
                "бетон": ["М100", "М200", "М300", "B15", "B20", "B25", "ГОСТ 7473-2010"],
                "арматура": ["А500С", "А400", "А240", "ГОСТ 5781-82"],
                "щебень": ["5-20", "20-40", "40-70", "ГОСТ 8267-93"],
                "песок": ["речной", "карьерный", "мытый", "ГОСТ 8736-2014"]
            },
            "stations": [
                "Москва-Товарная", "Ферзиково", "Калуга-1", "Обнинск",
                "Санкт-Петербург-Товарный", "Новосибирск-Главный"
            ],
            "incoterms": [
                "EXW", "FCA", "CPT", "CIP", "DAT", "DAP", "DDP",
                "FAS", "FOB", "CFR", "CIF"
            ],
            "units": {
                "weight": ["кг", "т", "тонн", "килограмм"],
                "volume": ["м3", "куб.м", "литр", "л"],
                "pieces": ["шт", "штук", "единиц"],
                "length": ["м", "метр", "км", "мм"]
            },
            "departments": [
                "Отдел логистики", "Отдел закупок", "Юридический отдел",
                "Финансовый отдел", "Отдел продаж", "HR отдел"
            ]
        }

    def _init_llm(self):
        """Инициализация LLM модели"""
        try:
            if self.llm_model_path and Path(self.llm_model_path).exists():
                # Проверяем тип модели
                if self.llm_model_path.endswith('.gguf'):
                    # GGUF модель - используем llama-cpp-python
                    from llama_cpp import Llama
                    self.llm = Llama(
                        model_path=self.llm_model_path,
                        n_ctx=2048,
                        n_threads=8
                    )
                    self.llm_type = "gguf"
                else:
                    # Обычная модель - используем transformers
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import torch

                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.llm_model_path)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.llm_model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    self.llm_type = "transformers"
                logger.info(f"LLM модель загружена: {self.llm_model_path}")
            else:
                self.use_llm = False
                logger.warning(
                    "LLM модель не найдена, используем только правила")
        except Exception as e:
            logger.error(f"Ошибка загрузки LLM: {e}")
            self.use_llm = False

    def extract(self, query: str) -> ExtractedEntities:
        """
        Главный метод извлечения сущностей

        Args:
            query: Пользовательский запрос

        Returns:
            ExtractedEntities: Извлеченные сущности
        """
        entities = ExtractedEntities()

        # 1. Извлечение правилами и регулярками
        self._extract_with_rules(query, entities)

        # 2. Извлечение с помощью NER
        self._extract_with_ner(query, entities)

        # 3. Извлечение с помощью LLM (если доступна)
        if self.use_llm:
            self._extract_with_llm(query, entities)

        # 4. Нормализация и валидация
        self._normalize_entities(entities)

        # 5. Вычисление уверенности
        self._calculate_confidence(entities)

        return entities

    def _extract_with_rules(self, query: str, entities: ExtractedEntities):
        """Извлечение с помощью правил и регулярных выражений"""
        query_lower = query.lower()

        # Определение домена по ключевым словам
        domain_keywords = {
            Domain.LEGAL: ["договор", "контракт", "соглашение", "юридический", "правовой"],
            Domain.LOGISTICS: ["поставка", "доставка", "отгрузка", "транспорт", "логистика"],
            Domain.PROCUREMENT: ["закупка", "тендер", "конкурс", "поставщик", "снабжение"],
            Domain.PRODUCTION: ["производство", "цех", "линия", "изготовление", "выпуск"],
            Domain.FINANCE: ["оплата", "счет", "бюджет", "финансы", "расчет"],
            Domain.HR: ["сотрудник", "вакансия", "персонал", "кадры", "hr"]
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                entities.domain = domain
                break

        # Извлечение ГОСТ/ТУ
        gost_pattern = r'ГОСТ\s*[\d\-\.]+'
        tu_pattern = r'ТУ\s*[\d\-\.]+'

        gosts = re.findall(gost_pattern, query, re.IGNORECASE)
        tus = re.findall(tu_pattern, query, re.IGNORECASE)
        entities.object_attributes.extend(gosts + tus)

        # Извлечение дат
        date_patterns = [
            r'\d{1,2}[\.\/]\d{1,2}[\.\/]\d{2,4}',  # DD.MM.YYYY
            r'\d{4}[-\.]\d{1,2}[-\.]\d{1,2}',      # YYYY-MM-DD
            r'до\s+\d{1,2}\s+\w+\s+\d{4}',         # до 15 января 2025
        ]

        for pattern in date_patterns:
            dates = re.findall(pattern, query)
            if dates:
                if not entities.time:
                    entities.time = {}
                entities.time['dates_found'] = dates

        # Извлечение количеств
        quantity_pattern = r'(\d+(?:\.\d+)?)\s*(тонн|т|кг|м3|куб\.м|шт|штук)'
        quantities = re.findall(quantity_pattern, query, re.IGNORECASE)
        if quantities:
            value, unit = quantities[0]
            entities.quantity = {"value": float(value), "unit": unit}

        # Извлечение станций из справочника
        for station in self.dictionaries.get("stations", []):
            if station.lower() in query_lower:
                if not entities.location:
                    entities.location = {}
                if "станци" in query_lower[:query_lower.index(station.lower())]:
                    entities.location["destination"] = station
                else:
                    entities.location["origin"] = station

        # Извлечение Инкотермс
        for incoterm in self.dictionaries.get("incoterms", []):
            if incoterm in query.upper():
                entities.constraints.append(f"Инкотермс: {incoterm}")

        # Извлечение номеров документов
        doc_patterns = [
            r'№\s*[\d\-\/]+',
            r'договор\s+[\d\-\/]+',
            r'контракт\s+[\d\-\/]+',
        ]

        for pattern in doc_patterns:
            docs = re.findall(pattern, query, re.IGNORECASE)
            entities.identifiers.extend(docs)

    def _extract_with_ner(self, query: str, entities: ExtractedEntities):
        """Извлечение с помощью NER (Named Entity Recognition)"""
        try:
            # Попробуем использовать Natasha для русского языка
            from natasha import (
                Segmenter, MorphVocab,
                NewsEmbedding, NewsMorphTagger,
                NewsSyntaxParser, NewsNERTagger,
                PER, ORG, LOC, Doc
            )

            segmenter = Segmenter()
            morph_vocab = MorphVocab()
            emb = NewsEmbedding()
            morph_tagger = NewsMorphTagger(emb)
            syntax_parser = NewsSyntaxParser(emb)
            ner_tagger = NewsNERTagger(emb)

            doc = Doc(query)
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            doc.parse_syntax(syntax_parser)
            doc.tag_ner(ner_tagger)

            for span in doc.spans:
                span.normalize(morph_vocab)
                if span.type == ORG:
                    entities.actors.append(span.normal)
                elif span.type == LOC:
                    if not entities.location:
                        entities.location = {}
                    entities.location[span.normal] = span.normal
                elif span.type == PER:
                    entities.actors.append(span.normal)

        except ImportError:
            # Если Natasha не установлена, используем простые правила
            # Извлечение организаций по шаблонам
            org_patterns = [
                r'ООО\s+"[^"]+"',
                r'ООО\s+\w+',
                r'АО\s+"[^"]+"',
                r'АО\s+\w+',
                r'ПАО\s+"[^"]+"',
                r'ПАО\s+\w+',
            ]

            for pattern in org_patterns:
                orgs = re.findall(pattern, query)
                entities.actors.extend(orgs)

    def _extract_with_llm(self, query: str, entities: ExtractedEntities):
        """Извлечение с помощью LLM"""
        if not self.use_llm:
            return

        prompt = self._build_llm_prompt(query)

        try:
            if self.llm_type == "gguf":
                # Используем GGUF модель
                response = self.llm(
                    prompt,
                    max_tokens=500,
                    temperature=0.3,
                    top_p=0.9,
                    echo=False
                )
                llm_output = response['choices'][0]['text']
            else:
                # Используем transformers модель
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt")

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=500,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9
                    )

                llm_output = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True)
                # Убираем промпт из ответа
                llm_output = llm_output[len(prompt):].strip()

            # Парсим JSON из ответа LLM
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                llm_entities = json.loads(json_match.group())
                self._merge_llm_entities(entities, llm_entities)

        except Exception as e:
            logger.error(f"Ошибка при извлечении с LLM: {e}")

    def _build_llm_prompt(self, query: str) -> str:
        """Построение промпта для LLM"""
        return f"""Ты — парсер запросов. Извлеки сущности из текста и верни ТОЛЬКО валидный JSON.

Текст запроса: "{query}"

Верни JSON со следующими полями (если поле не найдено, используй null):
{{
    "intent": "цель/намерение запроса",
    "domain": "область (Юристы/Логистика/Закупки/Производство/Финансы/HR)",
    "object": "объект запроса (товар/услуга/документ/процесс)",
    "action": "действие (поиск/сравнение/проверка/генерация/расчет)",
    "object_attributes": ["список атрибутов объекта"],
    "identifiers": ["идентификаторы и номера документов"],
    "actors": ["участники и организации"],
    "location": {{"origin": "откуда", "destination": "куда"}},
    "time": {{"from": "дата начала", "to": "дата конца", "deadline": "крайний срок"}},
    "quantity": {{"value": число, "unit": "единица измерения"}},
    "price": {{"amount": число, "currency": "валюта", "vat": "НДС"}},
    "constraints": ["ограничения и условия"],
    "exclusions": ["исключения"]
}}

Правила:
- Даты в формате ISO 8601 (YYYY-MM-DD)
- Числа без кавычек
- Не придумывай значения
- Используй null для отсутствующих полей

JSON:"""

    def _merge_llm_entities(self, entities: ExtractedEntities, llm_entities: Dict[str, Any]):
        """Объединение результатов LLM с существующими сущностями"""
        # Приоритет отдаем LLM для сложных полей
        if llm_entities.get("intent") and not entities.intent:
            entities.intent = llm_entities["intent"]

        if llm_entities.get("domain"):
            try:
                entities.domain = Domain(llm_entities["domain"])
            except ValueError:
                pass

        # Объединяем списки
        for field in ["object_attributes", "identifiers", "actors", "constraints", "exclusions"]:
            if llm_entities.get(field):
                existing = getattr(entities, field, [])
                new_items = llm_entities[field]
                if isinstance(new_items, list):
                    # Добавляем только уникальные элементы
                    for item in new_items:
                        if item not in existing:
                            existing.append(item)

        # Обновляем словари если они более полные
        for field in ["location", "time", "quantity", "price"]:
            if llm_entities.get(field) and isinstance(llm_entities[field], dict):
                if not getattr(entities, field):
                    setattr(entities, field, llm_entities[field])
                else:
                    # Объединяем словари
                    existing = getattr(entities, field)
                    existing.update(llm_entities[field])

    def _normalize_entities(self, entities: ExtractedEntities):
        """Нормализация и приведение к стандартному виду"""
        # Нормализация дат
        if entities.time and entities.time.get("dates_found"):
            dates = entities.time["dates_found"]
            # Простая нормализация - можно улучшить
            entities.time = {"raw_dates": dates}

        # Нормализация единиц измерения
        if entities.quantity:
            unit = entities.quantity.get("unit", "")
            unit_mapping = {
                "т": "тонн",
                "кг": "килограмм",
                "шт": "штук",
                "куб.м": "м3"
            }
            if unit in unit_mapping:
                entities.quantity["unit"] = unit_mapping[unit]

        # Удаление дубликатов
        entities.object_attributes = list(set(entities.object_attributes))
        entities.identifiers = list(set(entities.identifiers))
        entities.actors = list(set(entities.actors))
        entities.constraints = list(set(entities.constraints))
        entities.exclusions = list(set(entities.exclusions))

    def _calculate_confidence(self, entities: ExtractedEntities):
        """Вычисление уверенности в извлеченных сущностях"""
        # Простая эвристика для оценки уверенности
        confidences = {}

        if entities.domain:
            confidences["domain"] = 0.9 if entities.domain != Domain.UNKNOWN else 0.3

        if entities.object_type:
            confidences["object_type"] = 0.85

        if entities.quantity:
            confidences["quantity"] = 0.95  # Числа обычно извлекаются точно

        if entities.location:
            confidences["location"] = 0.8

        if entities.actors:
            confidences["actors"] = 0.7 if len(entities.actors) > 0 else 0.0

        entities.confidences = confidences


# Функция для быстрого тестирования
def test_extractor():
    """Тестирование экстрактора"""
    extractor = EntityExtractor(use_llm=False)  # Без LLM для быстрого теста

    test_queries = [
        "Нужен договор поставки цемента М500 ГОСТ 31108-2020 в количестве 3000 тонн на станцию Ферзиково",
        "Найти все счета от ООО Стройматериалы за последний квартал 2024 года",
        "Сравнить условия поставки щебня фракции 20-40 по железной дороге FCA Москва-Товарная",
        "Рассчитать стоимость доставки 500 тонн арматуры А500С до склада в Калуге",
    ]

    for query in test_queries:
        print(f"\nЗапрос: {query}")
        print("-" * 50)
        entities = extractor.extract(query)
        print(json.dumps(entities.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_extractor()
