# RAG ULTRA v0.1 - Инструкция по использованию

## 🎯 Обзор

RAG ULTRA - это система для создания и использования Retrieval-Augmented Generation (RAG) на основе корпоративных документов. Система поддерживает работу с различными форматами документов и предоставляет семантический поиск.

## ✅ Исправленные проблемы

Все основные проблемы в системе были исправлены:

1. **Зависимости** - установлены все необходимые пакеты
2. **Пути к моделям** - автоматическое исправление путей в зависимости от директории запуска
3. **Конфигурация** - добавлены недостающие параметры
4. **Обработка ошибок** - улучшена обработка ошибок и валидация

## 🚀 Быстрый старт

### 1. Подготовка окружения

```bash
# Активируйте окружение corp-rag
conda activate corp-rag

# Убедитесь, что вы в корневой директории проекта
cd /path/to/RAG_ULTRA_v0.1-coursor
```

### 2. Подготовка документов

```bash
# Создайте папку для документов (если её нет)
mkdir -p doc

# Поместите ваши документы в папку doc/
# Поддерживаемые форматы: PDF, DOCX, PPTX, XLSX, CSV, TXT, HTML, MD
```

### 3. Запуск полного пайплайна

```bash
# Запустите полный пайплайн одним командой
python run_pipeline.py
```

Этот скрипт автоматически выполнит все этапы:
1. Парсинг документов
2. Построение индексов полей
3. Создание лексикона
4. Создание эмбеддингов

## 📁 Структура файлов

После выполнения пайплайна создаются следующие файлы:

```
outputs/
├── dataset/
│   └── elements.jsonl          # Элементы документов
├── index/
│   ├── faiss.index            # FAISS индекс для поиска
│   ├── meta.jsonl             # Метаданные документов
│   ├── doc_fields_meta.json   # Метаданные полей документов
│   ├── doc_fields.npz         # Эмбеддинги полей документов
│   └── lexicon_doc_fields.json # Лексикон полей
├── graphs/
│   └── corpus_graph.*         # Граф документов
└── media/                     # Медиафайлы (изображения, формулы)
```

## 🌐 API сервис

### Запуск API сервиса

```bash
# Запуск API сервиса из корневой директории
python run_api.py

# Или запуск напрямую
uvicorn doc_ingest.service.minimal_search_api:app --host 0.0.0.0 --port 8000

# API доступен: http://localhost:8000
# Документация: http://localhost:8000/docs
# Проверка здоровья: http://localhost:8000/healthz
```

**Примечание:** API сервис автоматически исправляет пути к моделям и работает из любой директории.

### Использование API

```bash
# Проверка здоровья сервиса
curl http://localhost:8000/healthz

# Поиск документов
curl "http://localhost:8000/search?q=ваш_запрос&top_k=5"
```

## 🔍 Использование поиска

### Простой поиск

```python
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path.cwd() / "doc_ingest"))

from retrieval.searcher import SemanticSearcher

# Инициализация поисковика
searcher = SemanticSearcher()

# Поиск
results = searcher.search_raw('ваш запрос', top_k=5)

# Вывод результатов
for doc_result in results.results:
    print(f"Документ: {doc_result.doc_path}")
    print(f"Скор: {doc_result.best_score:.3f}")
    print(f"Контекст: {doc_result.context[:200]}...")
    print()
```

### Расширенный поиск

```python
from retrieval.universal_searcher import UniversalSearcher, UniversalSearchConfig
from retrieval.reranker import HierarchicalReranker, RerankerConfig

# Настройка поисковика
search_config = UniversalSearchConfig(
    semantic_top_k=100,
    top_k_docs=5,
    max_chunks_per_doc=4
)

reranker_config = RerankerConfig(
    kind="hybrid",
    model_path="doc_ingest/models/reranker_ru"
)

# Создание универсального поисковика
universal_searcher = UniversalSearcher(
    base_searcher=searcher,
    cfg=search_config,
    final_reranker=reranker
)

# Поиск с реранкингом
results = universal_searcher.search('ваш запрос')
```

## 🛠️ Ручной запуск этапов

Если нужно запустить отдельные этапы:

```bash
# 1. Парсинг документов
python doc_ingest/1.main_parse_doc.py

# 2. Построение индексов полей
python doc_ingest/2.main_build_doc_fields_index.py

# 3. Создание лексикона
python doc_ingest/3.main_build_field_lexicon.py

# 4. Создание эмбеддингов
python doc_ingest/4.main_embed.py
```

## 🔧 Конфигурация

Основные настройки находятся в `doc_ingest/config.py`:

- **EmbeddingConfig** - настройки эмбеддингов
- **RerankerConfig** - настройки реранкера
- **ContextConfig** - настройки контекста
- **PipelineConfig** - настройки пайплайна

## 📋 Поддерживаемые форматы

- **PDF** - с поддержкой OCR и извлечения таблиц
- **DOCX** - документы Word
- **PPTX** - презентации PowerPoint
- **XLSX/XLS** - таблицы Excel
- **CSV** - CSV файлы
- **TXT** - текстовые файлы
- **HTML/HTM** - веб-страницы
- **MD** - Markdown файлы

## ⚠️ Важные замечания

1. **Окружение** - обязательно используйте окружение `corp-rag`
2. **Пути** - скрипты автоматически исправляют пути к моделям
3. **Модели** - локальные модели должны быть в папке `doc_ingest/models/`
4. **Qdrant** - опционально, предупреждение о недоступности нормально

## 🐛 Устранение неполадок

### Ошибка "Path not found"
- Убедитесь, что модели находятся в правильных папках
- Проверьте, что запускаете скрипты из правильной директории

### Ошибка импорта
- Активируйте окружение `corp-rag`
- Убедитесь, что все зависимости установлены

### Пустая папка doc
- Создайте папку `doc` и поместите туда документы
- Проверьте, что документы имеют поддерживаемые расширения

## 📞 Поддержка

При возникновении проблем:
1. Проверьте, что используете правильное окружение
2. Убедитесь, что все файлы на месте
3. Проверьте логи выполнения скриптов

---

**RAG ULTRA v0.1 готов к использованию!** 🎉
