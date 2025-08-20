# Оптимизации проекта RAG_ULTRA

## Обзор улучшений

Проект был полностью оптимизирован для повышения производительности и улучшения качества реранкинга с учетом иерархии документов.

## 🚀 Основные оптимизации производительности

### 1. Кэширование результатов
- **LRU кэш** для эмбеддингов запросов (размер: 1000)
- **Кэш результатов** реранкера (размер: 5000)
- **Кэш поисковика** (размер: 1000)
- **Статистика попаданий** для мониторинга эффективности

### 2. Адаптивный батчинг
- **Динамический размер батча** для CrossEncoder (4-16)
- **Оптимизированные размеры** для разных моделей
- **Параллельная обработка** с ThreadPoolExecutor

### 3. Асинхронность
- **Многопоточность** для реранкера (max_workers: 2)
- **Неблокирующие операции** для I/O
- **Эффективное использование ресурсов**

### 4. Оптимизация памяти
- **Умное усечение** текста по токенам
- **Ограничение кандидатов** (max_candidates_per_query: 100)
- **Early stopping** для экономии вычислений

## 🎯 Улучшения реранкера

### 1. Гибридный режим
```python
RerankerConfig(
    kind="hybrid",  # CrossEncoder + Cosine
    hybrid_weights={"cross_encoder": 0.7, "cosine": 0.3}
)
```

### 2. Учет иерархии документов
- **Heading weight**: 0.8 (важность заголовков)
- **Path weight**: 0.6 (важность пути)
- **Category weight**: 0.4 (важность категории)
- **Structural coherence**: анализ связности страниц

### 3. Адаптивные алгоритмы
- **Confidence threshold**: 0.3 (порог уверенности)
- **Early stopping**: остановка при достижении качества
- **Smart truncation**: умное усечение текста

## 🏗️ Улучшения контекст-билдера

### 1. Сохранение иерархии
```python
ContextConfig(
    preserve_hierarchy=True,
    group_by_sections=True,
    min_section_coverage=0.7
)
```

### 2. Умное группирование
- **Группировка по разделам** (1.2.3, 2.1, etc.)
- **Сохранение структурной связности**
- **Анализ покрытия тематики**

### 3. Качество контекста
- **Метрики качества**: разнообразие источников, покрытие иерархии
- **Умное усечение** с сохранением предложений
- **Анализ релевантности** по разделам

## 📊 Графовый анализатор

### 1. Иерархическая структура
- **Анализ номеров разделов** (1.2.3, Раздел 1, etc.)
- **Вычисление важности** разделов
- **Семантические связи** между разделами

### 2. Метрики графа
- **Глубина иерархии**
- **Важность разделов**
- **Связность структуры**

## ⚡ Конфигурация производительности

### EmbeddingConfig
```python
@dataclass
class EmbeddingConfig:
    batch_size: int = 32
    max_concurrent_batches: int = 4
    enable_cache: bool = True
    cache_size: int = 10000
```

### RerankerConfig
```python
@dataclass
class RerankerConfig:
    batch_size: int = 8
    adaptive_batching: bool = True
    max_batch_size: int = 16
    min_batch_size: int = 4
    async_processing: bool = True
    max_workers: int = 2
```

## 🔧 Использование оптимизаций

### 1. Запуск оптимизированной системы
```bash
python main_optimized.py
```

### 2. Мониторинг производительности
```python
# Статистика кэша
rer_stats = rer.get_cache_stats()
use_stats = use.get_cache_stats()
base_stats = base.get_cache_stats()

# Очистка кэша
rer.clear_cache()
use.clear_cache()
base.clear_cache()
```

### 3. Настройка параметров
```python
# Адаптивные настройки
reranker_config = RerankerConfig(
    kind="hybrid",
    enable_cache=True,
    adaptive_batching=True,
    hierarchy_aware=True
)
```

## 📈 Ожидаемые улучшения

### Производительность
- **Ускорение поиска**: 2-5x (благодаря кэшированию)
- **Ускорение реранкинга**: 3-8x (адаптивный батчинг)
- **Снижение памяти**: 20-40% (умное усечение)

### Качество
- **Лучшая релевантность**: учет иерархии
- **Структурированный контекст**: группировка по разделам
- **Связность результатов**: анализ структурной связности

## 🚨 Важные замечания

### 1. Совместимость
- Все изменения **обратно совместимы**
- Старые импорты продолжают работать
- Новые функции опциональны

### 2. Требования
- Python 3.8+
- sentence-transformers
- networkx
- numpy, faiss

### 3. Мониторинг
- Следите за статистикой кэша
- Периодически очищайте кэш
- Адаптируйте размеры под ваши данные

## 🔍 Отладка и диагностика

### 1. Логирование
```python
# Включить детальное логирование
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Профилирование
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... ваш код ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats()
```

### 3. Анализ памяти
```python
import tracemalloc
tracemalloc.start()
# ... ваш код ...
current, peak = tracemalloc.get_traced_memory()
print(f"Текущее использование: {current / 1024 / 1024:.1f} MB")
print(f"Пиковое использование: {peak / 1024 / 1024:.1f} MB")
```

## 📝 Примеры использования

### Базовый поиск
```python
from retrieval.searcher import SemanticSearcher
from retrieval.reranker import HierarchicalReranker

searcher = SemanticSearcher()
reranker = HierarchicalReranker(RerankerConfig())

# Поиск с реранкингом
results = searcher.search("ваш запрос")
reranked = reranker.rerank("ваш запрос", results)
```

### Контекст-билдер
```python
from retrieval.context_builder import HierarchicalContextBuilder

builder = HierarchicalContextBuilder(ContextConfig())
context = builder.build(reranked)

print(f"Качество: {builder.get_quality_metrics(context)}")
```

### Графовый анализ
```python
from graph_builder import HierarchicalGraphBuilder

graph_builder = HierarchicalGraphBuilder()
graph = graph_builder.build_graph(elements)
analysis = graph_builder.analyze_hierarchy()
```

## 🎉 Заключение

Проект теперь включает:
- ✅ **Оптимизированную производительность**
- ✅ **Учет иерархии документов**
- ✅ **Улучшенный реранкер**
- ✅ **Структурированный контекст**
- ✅ **Графовый анализ**
- ✅ **Полную обратную совместимость**

Все оптимизации протестированы и готовы к использованию в продакшене.
