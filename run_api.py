#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска API сервиса RAG системы
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Запуск API сервиса"""
    print("🚀 ЗАПУСК API СЕРВИСА RAG СИСТЕМЫ")
    print("="*50)

    # Проверяем наличие необходимых файлов
    required_files = [
        "doc_ingest/outputs/index/faiss.index",
        "doc_ingest/outputs/index/meta.jsonl"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Отсутствуют необходимые файлы:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Сначала запустите полный пайплайн:")
        print("   python run_pipeline.py")
        sys.exit(1)

    print("✅ Все необходимые файлы найдены")
    print("\n🌐 Запуск API сервиса...")
    print("   API будет доступен по адресу: http://localhost:8000")
    print("   Документация: http://localhost:8000/docs")
    print("\n⏹️  Для остановки нажмите Ctrl+C")
    print("-" * 50)

    try:
        # Запускаем API сервис
        cmd = [
            "uvicorn",
            "doc_ingest.service.minimal_search_api:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]

        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\n\n🛑 API сервис остановлен")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка запуска API сервиса: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ uvicorn не найден. Установите его:")
        print("   pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
