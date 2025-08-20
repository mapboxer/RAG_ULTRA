# -*- coding: utf-8 -*-
"""
Утилиты для работы с путями к моделям
"""

from pathlib import Path
from config import EmbeddingConfig, RerankerConfig


def fix_model_paths(cfg: EmbeddingConfig) -> EmbeddingConfig:
    """
    Автоматически исправляет пути к моделям в зависимости от того,
    откуда запускается скрипт
    """
    current_dir = Path.cwd()

    if current_dir.name == "doc_ingest":
        # Если запускаем из doc_ingest, используем относительные пути
        cfg.model_name = "models/sbert_large_nlu_ru"
    else:
        # Если запускаем из корневой директории, используем полные пути
        cfg.model_name = "doc_ingest/models/sbert_large_nlu_ru"

    return cfg


def fix_reranker_paths(cfg: RerankerConfig) -> RerankerConfig:
    """
    Автоматически исправляет пути к моделям реранкера
    """
    current_dir = Path.cwd()

    if current_dir.name == "doc_ingest":
        # Если запускаем из doc_ingest, используем относительные пути
        if cfg.model_path:
            cfg.model_path = "models/reranker_ru"
    else:
        # Если запускаем из корневой директории, используем полные пути
        if cfg.model_path:
            cfg.model_path = "doc_ingest/models/reranker_ru"

    return cfg
