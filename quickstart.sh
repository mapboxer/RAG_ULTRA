#!/bin/bash

# RAG ULTRA v0.1 - Скрипт быстрого запуска
# Использование: bash quickstart.sh

set -e  # Остановка при ошибке

echo "🚀 RAG ULTRA v0.1 - Установка и настройка"
echo "=========================================="
echo ""

# Проверка наличия conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda не найдена! Пожалуйста, установите Anaconda или Miniconda"
    echo "   Скачать: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Проверка существования окружения
if conda env list | grep -q "corp-rag"; then
    echo "✅ Окружение corp-rag уже существует"
    echo "   Активация окружения..."
    eval "$(conda shell.bash hook)"
    conda activate corp-rag
else
    echo "📦 Создание conda окружения..."
    conda env create -f environment.yml
    eval "$(conda shell.bash hook)"
    conda activate corp-rag
    echo "✅ Окружение создано и активировано"
fi

# Проверка Python и основных библиотек
echo ""
echo "🔍 Проверка установки..."
python -c "import torch, faiss, numpy; print('✅ Основные библиотеки установлены')" || {
    echo "❌ Ошибка импорта библиотек. Попробуйте переустановить окружение."
    exit 1
}

# Создание необходимых директорий
echo ""
echo "📁 Создание структуры проекта..."
mkdir -p doc_ingest/models
mkdir -p doc_ingest/doc/{PDF,WORD,EXCEL,PPTX,TXT_MD}
mkdir -p doc_ingest/outputs/{dataset,index,media,graphs}
echo "✅ Структура директорий создана"

# Проверка наличия моделей
echo ""
echo "🤖 Проверка наличия моделей..."

EMBED_MODEL="doc_ingest/models/sbert_large_nlu_ru"
RERANK_MODEL="doc_ingest/models/reranker_ru"

if [ -d "$EMBED_MODEL" ] && [ -f "$EMBED_MODEL/model.safetensors" ]; then
    echo "✅ Модель эмбеддингов найдена"
else
    echo "📥 Загрузка модели эмбеддингов (это может занять несколько минут)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ai-forever/sbert_large_nlu_ru',
    repo_type='model',
    local_dir='$EMBED_MODEL',
    local_dir_use_symlinks=False
)
print('✅ Модель эмбеддингов загружена')
"
fi

if [ -d "$RERANK_MODEL" ] && [ -f "$RERANK_MODEL/model.safetensors" ]; then
    echo "✅ Модель ранжирования найдена"
else
    echo "📥 Загрузка модели ранжирования..."
    python -c "
from sentence_transformers import CrossEncoder
ce = CrossEncoder('DiTy/cross-encoder-russian-msmarco')
ce.save_pretrained('$RERANK_MODEL')
print('✅ Модель ранжирования загружена')
"
fi

# Проверка наличия документов
echo ""
echo "📄 Проверка наличия документов..."
DOC_COUNT=$(find doc_ingest/doc -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.xlsx" -o -name "*.pptx" -o -name "*.txt" \) 2>/dev/null | wc -l)

if [ "$DOC_COUNT" -eq 0 ]; then
    echo "⚠️  Документы не найдены в doc_ingest/doc/"
    echo "   Пожалуйста, добавьте документы в соответствующие папки:"
    echo "   - PDF файлы в doc_ingest/doc/PDF/"
    echo "   - Word файлы в doc_ingest/doc/WORD/"
    echo "   - Excel файлы в doc_ingest/doc/EXCEL/"
    echo "   - PowerPoint файлы в doc_ingest/doc/PPTX/"
    echo "   - Текстовые файлы в doc_ingest/doc/TXT_MD/"
    echo ""
    read -p "Продолжить без документов? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    echo "✅ Найдено документов: $DOC_COUNT"
fi

# Запуск обработки
echo ""
echo "🔄 Запуск обработки документов..."
echo "================================"

cd doc_ingest

# Проверка существования выходных файлов
if [ -f "outputs/dataset/elements.jsonl" ]; then
    echo "⚠️  Найдены существующие данные обработки"
    read -p "Перезаписать? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Пропуск обработки документов..."
    else
        echo "1️⃣  Парсинг документов..."
        python 1.main_parse_doc.py
        
        echo "2️⃣  Построение индекса полей..."
        python 2.main_build_doc_fields_index.py
        
        echo "3️⃣  Создание лексикона..."
        python 3.main_build_field_lexicon.py
        
        echo "4️⃣  Векторизация и индексация..."
        python 4.main_embed.py
    fi
else
    if [ "$DOC_COUNT" -gt 0 ]; then
        echo "1️⃣  Парсинг документов..."
        python 1.main_parse_doc.py
        
        echo "2️⃣  Построение индекса полей..."
        python 2.main_build_doc_fields_index.py
        
        echo "3️⃣  Создание лексикона..."
        python 3.main_build_field_lexicon.py
        
        echo "4️⃣  Векторизация и индексация..."
        python 4.main_embed.py
    else
        echo "⚠️  Пропуск обработки - нет документов"
    fi
fi

cd ..

# Финальная информация
echo ""
echo "✨ Установка завершена!"
echo "======================="
echo ""
echo "📝 Следующие шаги:"
echo ""
echo "1. Для запуска API сервера:"
echo "   conda activate corp-rag"
echo "   cd doc_ingest/service"
echo "   uvicorn search_api:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "2. API будет доступен по адресу:"
echo "   http://localhost:8000"
echo "   Документация: http://localhost:8000/docs"
echo ""
echo "3. Для тестирования системы:"
echo "   cd doc_ingest"
echo "   python test_rag_system.py"
echo ""
echo "📚 Подробная документация: README.md"
