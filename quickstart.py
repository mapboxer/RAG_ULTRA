#!/usr/bin/env python3
"""
RAG ULTRA v0.1 - Скрипт быстрого запуска
Кроссплатформенная версия (Windows/Linux/MacOS)
Использование: python quickstart.py
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def run_command(cmd, shell=True, check=True):
    """Выполнение команды в терминале"""
    try:
        result = subprocess.run(
            cmd, shell=shell, check=check, capture_output=True, text=True)
        return result.returncode == 0, result.stdout
    except subprocess.CalledProcessError as e:
        return False, str(e)


def check_conda():
    """Проверка наличия conda"""
    return shutil.which('conda') is not None


def check_environment():
    """Проверка существования окружения corp-rag"""
    success, output = run_command("conda env list")
    return success and "corp-rag" in output


def create_directories():
    """Создание структуры директорий"""
    dirs = [
        "doc_ingest/models",
        "doc_ingest/doc/PDF",
        "doc_ingest/doc/WORD",
        "doc_ingest/doc/EXCEL",
        "doc_ingest/doc/PPTX",
        "doc_ingest/doc/TXT_MD",
        "doc_ingest/outputs/dataset",
        "doc_ingest/outputs/index",
        "doc_ingest/outputs/media",
        "doc_ingest/outputs/graphs"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return True


def check_models():
    """Проверка наличия моделей"""
    embed_model = Path(
        "doc_ingest/models/sbert_large_nlu_ru/model.safetensors")
    rerank_model = Path("doc_ingest/models/reranker_ru/model.safetensors")

    return embed_model.exists(), rerank_model.exists()


def download_models():
    """Загрузка моделей если их нет"""
    embed_exists, rerank_exists = check_models()

    if not embed_exists:
        print("📥 Загрузка модели эмбеддингов (это может занять несколько минут)...")
        code = """
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ai-forever/sbert_large_nlu_ru',
    repo_type='model',
    local_dir='doc_ingest/models/sbert_large_nlu_ru',
    local_dir_use_symlinks=False
)
print('✅ Модель эмбеддингов загружена')
"""
        try:
            exec(code)
        except Exception as e:
            print(f"❌ Ошибка загрузки модели эмбеддингов: {e}")
            return False
    else:
        print("✅ Модель эмбеддингов найдена")

    if not rerank_exists:
        print("📥 Загрузка модели ранжирования...")
        code = """
from sentence_transformers import CrossEncoder
ce = CrossEncoder('DiTy/cross-encoder-russian-msmarco')
ce.save_pretrained('doc_ingest/models/reranker_ru')
print('✅ Модель ранжирования загружена')
"""
        try:
            exec(code)
        except Exception as e:
            print(f"❌ Ошибка загрузки модели ранжирования: {e}")
            return False
    else:
        print("✅ Модель ранжирования найдена")

    return True


def count_documents():
    """Подсчет документов"""
    extensions = ['.pdf', '.docx', '.doc', '.xlsx',
                  '.xls', '.pptx', '.txt', '.md', '.html']
    doc_dir = Path("doc_ingest/doc")

    if not doc_dir.exists():
        return 0

    count = 0
    for ext in extensions:
        count += len(list(doc_dir.rglob(f"*{ext}")))

    return count


def process_documents():
    """Запуск обработки документов"""
    scripts = [
        ("1️⃣  Парсинг документов...", "1.main_parse_doc.py"),
        ("2️⃣  Построение индекса полей...", "2.main_build_doc_fields_index.py"),
        ("3️⃣  Создание лексикона...", "3.main_build_field_lexicon.py"),
        ("4️⃣  Векторизация и индексация...", "4.main_embed.py")
    ]

    os.chdir("doc_ingest")

    for message, script in scripts:
        print(message)
        success, _ = run_command(f"python {script}")
        if not success:
            print(f"❌ Ошибка при выполнении {script}")
            os.chdir("..")
            return False

    os.chdir("..")
    return True


def main():
    print("🚀 RAG ULTRA v0.1 - Установка и настройка")
    print("==========================================")
    print()

    # Проверка conda
    if not check_conda():
        print("❌ Conda не найдена! Пожалуйста, установите Anaconda или Miniconda")
        print("   Скачать: https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)

    # Проверка/создание окружения
    if check_environment():
        print("✅ Окружение corp-rag уже существует")
    else:
        print("📦 Создание conda окружения...")
        success, _ = run_command("conda env create -f environment.yml")
        if not success:
            print("❌ Ошибка создания окружения")
            sys.exit(1)
        print("✅ Окружение создано")

    print("\n⚠️  Активируйте окружение командой:")
    print("   conda activate corp-rag")
    print("   Затем запустите скрипт снова\n")

    # Проверка активации окружения
    try:
        import torch
        import faiss
        import numpy
        print("✅ Основные библиотеки доступны")
    except ImportError as e:
        print(f"⚠️  Некоторые библиотеки не установлены: {e}")
        print("   Убедитесь, что окружение corp-rag активировано")
        response = input("Продолжить? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    # Создание директорий
    print("\n📁 Создание структуры проекта...")
    if create_directories():
        print("✅ Структура директорий создана")

    # Проверка и загрузка моделей
    print("\n🤖 Проверка наличия моделей...")
    if not download_models():
        print("⚠️  Не все модели загружены, но можно продолжить")

    # Проверка документов
    print("\n📄 Проверка наличия документов...")
    doc_count = count_documents()

    if doc_count == 0:
        print("⚠️  Документы не найдены в doc_ingest/doc/")
        print("   Пожалуйста, добавьте документы в соответствующие папки:")
        print("   - PDF файлы в doc_ingest/doc/PDF/")
        print("   - Word файлы в doc_ingest/doc/WORD/")
        print("   - Excel файлы в doc_ingest/doc/EXCEL/")
        print("   - PowerPoint файлы в doc_ingest/doc/PPTX/")
        print("   - Текстовые файлы в doc_ingest/doc/TXT_MD/")
        response = input("\nПродолжить без документов? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    else:
        print(f"✅ Найдено документов: {doc_count}")

    # Обработка документов
    if Path("doc_ingest/outputs/dataset/elements.jsonl").exists():
        print("\n⚠️  Найдены существующие данные обработки")
        response = input("Перезаписать? (y/n): ")
        if response.lower() == 'y' and doc_count > 0:
            print("\n🔄 Запуск обработки документов...")
            print("================================")
            if process_documents():
                print("✅ Обработка завершена успешно")
            else:
                print("⚠️  Обработка завершена с ошибками")
    elif doc_count > 0:
        print("\n🔄 Запуск обработки документов...")
        print("================================")
        if process_documents():
            print("✅ Обработка завершена успешно")
        else:
            print("⚠️  Обработка завершена с ошибками")

    # Финальная информация
    print("\n✨ Установка завершена!")
    print("=======================")
    print("\n📝 Следующие шаги:")
    print("\n1. Для запуска API сервера:")
    print("   conda activate corp-rag")
    print("   cd doc_ingest/service")
    print("   uvicorn search_api:app --reload --host 0.0.0.0 --port 8000")
    print("\n2. API будет доступен по адресу:")
    print("   http://localhost:8000")
    print("   Документация: http://localhost:8000/docs")
    print("\n3. Для тестирования системы:")
    print("   cd doc_ingest")
    print("   python test_rag_system.py")
    print("\n📚 Подробная документация: README.md")


if __name__ == "__main__":
    main()
