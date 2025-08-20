# config.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    # входы (без дефолтов — первыми)
    categories: Dict[str, List[str]]  # {"Категория": ["doc/..pdf", ...], ...}

    # postgres как источник (опционально)
    # "select id, title, content from documents"
    pg_query: Optional[str] = None
    pg_dsn: Optional[str] = None  # строка подключения к PostgreSQL

    # парсинг
    pdf_ocr_fallback: bool = True     # OCR для сканов
    pdf_table_extraction: bool = True  # попытка вытаскивать таблицы
    keep_images: bool = True          # сохранять изображения
    keep_formulas: bool = True        # <-- было 'keep_formulas: True' (ошибка)

    # chunk/порядок
    max_chars_per_element: int = 15000

    # производительность
    workers: int = 4


@dataclass
class EmbeddingConfig:
    # ЛОКАЛЬНАЯ модель (ваша)
    # абсолютный или относительный путь
    model_name: str = "doc_ingest/models/sbert_large_nlu_ru"
    device: str = "auto"                           # "cuda" | "cpu" | "auto"

    # SBERT не использует e5-префиксы
    use_e5_prefix: bool = False

    # оффлайн и локальные файлы
    offline_mode: bool = True
    local_files_only: bool = True

    # у sbert-папок токенайзер обычно лежит в подпапке 0_Transformer
    tokenizer_subfolder: Optional[str] = "0_Transformer"

    # чанкинг
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 512
    min_chunk_tokens: int = 64
    sentence_overlap: int = 1
    cohesion_split: bool = True
    heading_aware: bool = True
    table_as_is: bool = True

    # индексы
    index_kind: str = "both"   # "faiss" | "qdrant" | "both"
    faiss_out: str = "outputs/index/faiss.index"
    meta_out: str = "outputs/index/meta.jsonl"

    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    qdrant_collection: str = "cementum_docs"

    # Оптимизации производительности
    batch_size: int = 32
    max_concurrent_batches: int = 4
    enable_cache: bool = True
    cache_size: int = 10000


@dataclass
class RerankerConfig:
    kind: str = "hybrid"                # "none" | "cosine" | "cross_encoder" | "hybrid"
    # None | путь к локальной CrossEncoder-модели (если kind="cross_encoder"), например: "models/reranker_ru"
    model_path: Optional[str] = "doc_ingest/models/reranker_ru"
    batch_size: int = 8                        # Оптимизирован для CrossEncoder
    device: str = "auto"
    offline_mode: bool = True            # не ходить в сеть

    # Новые оптимизации
    enable_cache: bool = True
    cache_size: int = 5000
    adaptive_batching: bool = True
    max_batch_size: int = 16
    min_batch_size: int = 4
    confidence_threshold: float = 0.3
    early_stopping: bool = True
    max_candidates_per_query: int = 100

    # Гибридный режим
    # {"cross_encoder": 0.7, "cosine": 0.3}
    hybrid_weights: Dict[str, float] = None

    # Асинхронность
    async_processing: bool = True
    max_workers: int = 2

    # Учет иерархии
    hierarchy_aware: bool = True
    heading_weight: float = 0.8
    path_weight: float = 0.6
    category_weight: float = 0.4
    structural_coherence: bool = True


@dataclass
class ContextConfig:
    token_budget: int = 2000             # общий бюджет токенов под контекст
    per_doc_max_chunks: int = 5          # максимум чанков на документ
    cite_with_pages: bool = True
    include_headings: bool = True
    join_separator: str = "\n\n---\n\n"  # сепаратор между болюсами контекста

    # Оптимизации контекста
    relevance_threshold: float = 0.6
    max_context_length: int = 4000
    enable_smart_truncation: bool = True

    # Учет иерархии в контексте
    preserve_hierarchy: bool = True
    group_by_sections: bool = True
    min_section_coverage: float = 0.7

# @dataclass
# class EmbeddingConfig:
#     # модель эмбеддингов (по умолчанию — высокое качество, RU-friendly)
#     model_name: str = "intfloat/multilingual-e5-base"     # альтернатива: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#     device: str = "auto"                                  # "cuda", "cpu", "auto"

#     # правила семантического чанкинга
#     chunk_target_tokens: int = 350
#     chunk_max_tokens: int = 512
#     min_chunk_tokens: int = 64
#     sentence_overlap: int = 1
#     cohesion_split: bool = True          # разбиение по падению семантической связности
#     heading_aware: bool = True           # использовать заголовки/номера разделов
#     table_as_is: bool = True             # таблица — отдельный чанк, но с контекстом подписи/заголовка

#     # индексация
#     index_kind: str = "both"             # "faiss" | "qdrant" | "both"
#     faiss_out: str = "outputs/index/faiss.index"
#     meta_out: str = "outputs/index/meta.jsonl"

#     # Qdrant
#     qdrant_host: str = "127.0.0.1"
#     qdrant_port: int = 6333
#     qdrant_collection: str = "cementum_docs"
