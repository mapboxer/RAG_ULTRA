#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска полного пайплайна RAG системы
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Запускает команду и выводит результат"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Команда: {cmd}")

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Успешно выполнено!")
        if result.stdout:
            print("Вывод:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка выполнения:")
        print(f"Код ошибки: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def main():
    """Основная функция запуска пайплайна"""
    print("🎯 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА RAG СИСТЕМЫ")
    print("="*60)

    # Проверяем наличие папки doc
    doc_path = Path("doc")
    if not doc_path.exists():
        print(f"❌ Папка '{doc_path}' не найдена!")
        print("Создайте папку 'doc' и поместите туда документы для обработки.")
        sys.exit(1)

    # Список команд для выполнения
    commands = [
        ("python doc_ingest/1.main_parse_doc.py", "1. Парсинг документов"),
        ("python doc_ingest/2.main_build_doc_fields_index.py",
         "2. Построение индексов полей документов"),
        ("python doc_ingest/3.main_build_field_lexicon.py",
         "3. Создание лексикона полей"),
        ("python doc_ingest/4.main_embed.py", "4. Создание эмбеддингов и индексов")
    ]

    # Выполняем команды по порядку
    for cmd, description in commands:
        if not run_command(cmd, description):
            print(f"\n❌ Пайплайн остановлен на этапе: {description}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("🎉 ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    print(f"{'='*60}")
    print("📁 Созданные файлы:")
    print("   - outputs/dataset/elements.jsonl - элементы документов")
    print("   - outputs/index/faiss.index - FAISS индекс")
    print("   - outputs/index/meta.jsonl - метаданные")
    print("   - outputs/graphs/corpus_graph.* - граф документов")
    print("\n🔍 Для тестирования поиска используйте:")
    print("   python -c \"from doc_ingest.retrieval.searcher import SemanticSearcher; s=SemanticSearcher(); print(s.search_raw('тестовый запрос', top_k=3).results)\"")


if __name__ == "__main__":
    main()
