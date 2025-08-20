# -*- coding: utf-8 -*-
"""
FastAPI поисковый слой:
- Semantic top-k по FAISS
- Фильтры (category, doc_id, doc_path, heading_prefix)
- Группировка результатов по документу
- Окно вокруг чанка (±window соседей внутри документа)
"""

from retrieval.universal_searcher import UniversalSearcher, UniversalSearchConfig
from retrieval.searcher import SemanticSearcher
from embed_utils import load_embedding_stack
from config import EmbeddingConfig
from pydantic import BaseModel, Field
from fastapi import FastAPI, Query
import numpy as np
import faiss
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

# Добавляем путь к родительской директории для импорта модулей
sys.path.append(str(Path(__file__).parent.parent))


# Импорты с обработкой ошибок
try:
    from retrieval.reranker import Reranker
except ImportError:
    Reranker = None

try:
    from retrieval.graph_searcher import GraphSearcher
except ImportError:
    GraphSearcher = None

try:
    from entity_extractor import EntityExtractor
except ImportError:
    # Если entity_extractor недоступен, создаем заглушку
    class EntityExtractor:
        def __init__(self, use_llm=False):
            pass

        def extract_entities(self, text):
            return []


# ----------------- Модели ответов API -----------------

class SearchHit(BaseModel):
    score: float
    chunk_index: int
    text: str
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    heading_path: Optional[List[str]] = None
    element_types: Optional[List[str]] = None
    media_refs: Optional[List[str]] = None


class DocResult(BaseModel):
    doc_id: Optional[str] = None
    doc_path: Optional[str] = None
    category: Optional[str] = None
    best_score: float
    hits: List[SearchHit]
    # агрегированный контекст по документу (с учётом окна и без дублей)
    context: str
    page_span: Optional[Tuple[Optional[int], Optional[int]]] = None
    heading_preview: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    took_ms: int
    total_candidates: int
    results: List[DocResult]


# ----------------- Инициализация -----------------

app = FastAPI(title="Cementum Semantic Search API", version="0.1")

# Глобальное состояние (безопасно для чтения)
CFG = EmbeddingConfig()

# Исправляем пути к моделям для API сервиса
current_dir = Path.cwd()
if current_dir.name == "service":
    # Если запускаем из service, используем пути относительно doc_ingest
    CFG.model_name = "../models/sbert_large_nlu_ru"
    CFG.faiss_out = "../outputs/index/faiss.index"
    CFG.meta_out = "../outputs/index/meta.jsonl"
else:
    # Если запускаем из doc_ingest, используем относительные пути
    CFG.model_name = "models/sbert_large_nlu_ru"
    CFG.faiss_out = "outputs/index/faiss.index"
    CFG.meta_out = "outputs/index/meta.jsonl"

MODEL, TOKENIZER, PREFIX = load_embedding_stack(CFG)

FAISS_INDEX: Optional[faiss.Index] = None
# выровнено по порядку вектора в FAISS (i -> meta[i])
META: List[Dict[str, Any]] = []
# "category::doc_id|doc_path" -> [vec_idx... в порядке документа]
DOC2VECIDX: Dict[str, List[int]] = {}
VECIDX2DOCKEY: List[str] = []              # i -> doc_key
READY = False

# Новые компоненты для улучшенного поиска
UNIVERSAL_SEARCHER: Optional[UniversalSearcher] = None
GRAPH_SEARCHER: Optional[GraphSearcher] = None
ENTITY_EXTRACTOR: Optional[EntityExtractor] = None


def _doc_key(meta: Dict[str, Any]) -> str:
    cat = meta.get("category") or "UNK"
    did = meta.get("doc_id") or meta.get("doc_path") or "doc"
    return f"{cat}::{did}"


def _load_faiss_and_meta():
    global FAISS_INDEX, META, DOC2VECIDX, VECIDX2DOCKEY, READY

    # 1) FAISS
    FAISS_INDEX = faiss.read_index(CFG.faiss_out)

    # 2) META — предполагаем тот же порядок, что был при построении индекса
    META = []
    with open(CFG.meta_out, "r", encoding="utf-8") as f:
        for line in f:
            META.append(json.loads(line))

    if FAISS_INDEX.ntotal != len(META):
        raise RuntimeError(
            f"FAISS vectors ({FAISS_INDEX.ntotal}) and META rows ({len(META)}) "
            f"length mismatch. Rebuild index/meta to align."
        )

    # 3) Группировка по документам (для окна по чанкам)
    DOC2VECIDX = {}
    VECIDX2DOCKEY = [""] * len(META)
    # сортировка внутри дока по from_order/to_order (как стабильный порядок)
    for i, m in enumerate(META):
        dk = _doc_key(m)
        DOC2VECIDX.setdefault(dk, []).append(i)
        VECIDX2DOCKEY[i] = dk

    # сортируем позиции чанков внутри каждого документа
    for dk, idxs in DOC2VECIDX.items():
        idxs.sort(key=lambda ix: (META[ix].get(
            "from_order") or 0, META[ix].get("to_order") or 0))

    READY = True


@app.on_event("startup")
def _startup():
    global UNIVERSAL_SEARCHER, GRAPH_SEARCHER, ENTITY_EXTRACTOR

    _load_faiss_and_meta()

    # Инициализация новых компонентов
    try:
        # Создаем базовый семантический поисковик
        semantic_searcher = SemanticSearcher()

        # Создаем граф поисковик (если доступен)
        if GraphSearcher is not None:
            GRAPH_SEARCHER = GraphSearcher()
        else:
            GRAPH_SEARCHER = None
            print("⚠️ GraphSearcher недоступен")

        # Создаем экстрактор сущностей
        ENTITY_EXTRACTOR = EntityExtractor(use_llm=False)

        # Создаем реранкер если доступен
        reranker = None
        if Reranker is not None:
            try:
                reranker = Reranker()
            except Exception as e:
                print(f"⚠️ Reranker не доступен: {e}")
        else:
            print("⚠️ Reranker недоступен")

        # Создаем универсальный поисковик
        config = UniversalSearchConfig(
            use_graph_search=GRAPH_SEARCHER is not None,
            use_entity_extraction=True,
            graph_weight=0.3,
            entity_weight=0.2
        )

        UNIVERSAL_SEARCHER = UniversalSearcher(
            base_searcher=semantic_searcher,
            cfg=config,
            final_reranker=reranker,
            graph_searcher=GRAPH_SEARCHER,
            entity_extractor=ENTITY_EXTRACTOR
        )

        print("✅ Универсальный поисковик инициализирован")

    except Exception as e:
        print(f"⚠️ Не удалось инициализировать расширенный поиск: {e}")
        print("Используется базовый поиск")
        UNIVERSAL_SEARCHER = None


@app.get("/healthz")
def healthz():
    return {"status": "ok", "ready": READY, "vectors": FAISS_INDEX.ntotal if FAISS_INDEX else 0}


# ----------------- Вспомогательные -----------------

def _encode_query(text: str) -> np.ndarray:
    vec = MODEL.encode([PREFIX + text], convert_to_numpy=True,
                       normalize_embeddings=True)
    return vec.astype("float32")  # FAISS ожидает float32


def _filter_mask(
    meta: Dict[str, Any],
    categories: Optional[Set[str]],
    doc_ids: Optional[Set[str]],
    doc_paths: Optional[Set[str]],
    heading_prefixes: Optional[List[str]],
) -> bool:
    # category
    if categories:
        if (meta.get("category") or "UNK") not in categories:
            return False
    # doc_id
    if doc_ids:
        if (meta.get("doc_id") or "") not in doc_ids:
            return False
    # doc_path
    if doc_paths:
        if (meta.get("doc_path") or "") not in doc_paths:
            return False
    # heading_path prefix (любой из переданных должен быть префиксом)
    if heading_prefixes:
        hp = meta.get("heading_path") or []
        joined = " > ".join(hp) if hp else ""
        # нормализуем пробелы
        joined_norm = " ".join(joined.split())
        ok = False
        for prefix in heading_prefixes:
            pref_norm = " ".join((prefix or "").split())
            if joined_norm.startswith(pref_norm):
                ok = True
                break
        if not ok:
            return False
    return True


def _gather_doc_context(
    doc_key: str,
    center_vec_idx: int,
    window: int
) -> Tuple[List[int], str, Tuple[Optional[int], Optional[int]], Optional[str]]:
    """
    Возвращает:
    - список vec_idx (без дублей) в порядке документа, покрывающий окно вокруг центрального чанка;
    - склеенный текст контекста;
    - диапазон страниц (min_page, max_page);
    - короткий preview из heading_path (первые 1-2 заголовка).
    """
    doc_vecs = DOC2VECIDX.get(doc_key, [])
    if not doc_vecs:
        return [center_vec_idx], META[center_vec_idx]["text"], (META[center_vec_idx].get("page_from"), META[center_vec_idx].get("page_to")), None

    # позиция центрального индекса внутри документа
    try:
        pos = doc_vecs.index(center_vec_idx)
    except ValueError:
        # fallback
        return [center_vec_idx], META[center_vec_idx]["text"], (META[center_vec_idx].get("page_from"), META[center_vec_idx].get("page_to")), None

    start = max(0, pos - window)
    end = min(len(doc_vecs), pos + window + 1)
    span = doc_vecs[start:end]

    # текст и страницы
    parts, pages = [], []
    for ix in span:
        m = META[ix]
        parts.append(m.get("text") or "")
        if m.get("page_from") is not None:
            pages.append(m["page_from"])
        if m.get("page_to") is not None:
            pages.append(m["page_to"])
    context_text = "\n\n".join([p for p in parts if p.strip()])
    page_span = (min(pages) if pages else None, max(pages) if pages else None)

    # preview по заголовкам
    hp = META[center_vec_idx].get("heading_path") or []
    heading_preview = " > ".join(hp[:2]) if hp else None

    return span, context_text, page_span, heading_preview


# ----------------- Основной endpoint -----------------

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Поисковый запрос"),
    top_k: int = Query(10, ge=1, le=100),
    oversample: int = Query(
        5, ge=1, le=20, description="Во сколько раз превышать top_k при первичном отборе (для фильтров)"),
    window: int = Query(
        1, ge=0, le=5, description="Окно (±N) соседних чанков в том же документе"),
    category: Optional[List[str]] = Query(
        None, description="Фильтр по категориям (повторяй параметр)"),
    doc_id: Optional[List[str]] = Query(
        None, description="Фильтр по doc_id (повторяй параметр)"),
    doc_path: Optional[List[str]] = Query(
        None, description="Фильтр по doc_path (повторяй параметр)"),
    heading_prefix: Optional[List[str]] = Query(
        None, description="Фильтр по префиксу heading_path (строка)"),
):
    assert READY and FAISS_INDEX is not None

    t0 = time.time()

    # 1) Вектор запроса
    qvec = _encode_query(q)

    # 2) Поиск в FAISS (oversample для фильтров)
    k = min(FAISS_INDEX.ntotal, top_k * oversample)
    D, I = FAISS_INDEX.search(qvec, k)  # cosine via IP (вектора нормированы)
    scores = D[0]
    idxs = I[0]

    # 3) Применяем фильтры и группируем по документу
    cat_set = set(category) if category else None
    did_set = set(doc_id) if doc_id else None
    dpath_set = set(doc_path) if doc_path else None
    hp_list = heading_prefix if heading_prefix else None

    # doc_key -> {best_score, picks:set(vec_idx), hits:[...]}
    grouped: Dict[str, Dict[str, Any]] = {}
    picked_total = 0

    for ix, sc in zip(idxs, scores):
        if ix < 0:
            continue
        m = META[ix]
        if not _filter_mask(m, cat_set, did_set, dpath_set, hp_list):
            continue

        dk = _doc_key(m)
        # Вокруг текущего чанка собираем окно
        span, ctx, page_span, hp_prev = _gather_doc_context(
            dk, ix, window=window)

        # Сохраняем центральный хит (без дублей)
        hit = SearchHit(
            score=float(sc),
            chunk_index=int(m.get("chunk_index", ix)),
            text=m.get("text") or "",
            page_from=m.get("page_from"),
            page_to=m.get("page_to"),
            heading_path=m.get("heading_path") or None,
            element_types=m.get("element_types") or None,
            media_refs=m.get("media_refs") or None
        )

        g = grouped.setdefault(dk, {
            "best_score": float(sc),
            "picks": set(),     # type: Set[int]
            "hits": [],
            "doc_id": m.get("doc_id"),
            "doc_path": m.get("doc_path"),
            "category": m.get("category"),
            "contexts": [],     # для объединения нескольких окон (если есть)
            "pages": []
        })
        g["best_score"] = max(g["best_score"], float(sc))
        g["hits"].append(hit)
        g["picks"].update(span)
        if ctx:
            g["contexts"].append((span, ctx))
        if page_span and (page_span[0] is not None or page_span[1] is not None):
            g["pages"].append(page_span)

        picked_total += 1
        # ограничим по количеству документов (top_k) как только собрали достаточно
        if len(grouped) >= top_k and picked_total >= top_k * 3:  # небольшой гистерезис
            break

    # 4) Формируем результаты по документам и сортируем по лучшему скору
    results: List[DocResult] = []
    for dk, g in grouped.items():
        # Объединим несколько контекстных окон в один, сохраняя порядок документа
        picks_sorted = sorted(list(g["picks"]), key=lambda ix: (
            META[ix].get("from_order") or 0, META[ix].get("to_order") or 0))
        # Склеим без дублей
        parts = []
        seen: Set[int] = set()
        for ix in picks_sorted:
            if ix in seen:
                continue
            seen.add(ix)
            t = META[ix].get("text") or ""
            if t.strip():
                parts.append(t)
        context = "\n\n".join(parts)

        # агрегированный span страниц
        page_min, page_max = None, None
        for pmin, pmax in g["pages"]:
            if pmin is not None:
                page_min = pmin if page_min is None else min(page_min, pmin)
            if pmax is not None:
                page_max = pmax if page_max is None else max(page_max, pmax)

        # превью заголовков — по первому хиту
        heading_preview = None
        if g["hits"]:
            hp = g["hits"][0].heading_path or []
            heading_preview = " > ".join(hp[:2]) if hp else None

        results.append(DocResult(
            doc_id=g["doc_id"],
            doc_path=g["doc_path"],
            category=g["category"],
            best_score=g["best_score"],
            hits=g["hits"],
            context=context,
            page_span=(page_min, page_max),
            heading_preview=heading_preview
        ))

    results.sort(key=lambda r: r.best_score, reverse=True)
    took_ms = int((time.time() - t0) * 1000)

    return SearchResponse(
        query=q,
        took_ms=took_ms,
        total_candidates=int(k),
        results=results[:top_k]
    )


@app.get("/api/v2/search", response_model=SearchResponse)
def search_v2(
    q: str = Query(..., description="Поисковый запрос"),
    top_k: int = Query(5, ge=1, le=20, description="Количество документов"),
    use_graph: bool = Query(True, description="Использовать поиск по графу"),
    use_entities: bool = Query(
        True, description="Использовать извлечение сущностей"),
    use_reranker: bool = Query(
        True, description="Использовать переранжирование"),
    categories: Optional[str] = Query(
        None, description="Фильтр по категориям (через запятую)"),
):
    """
    Продвинутый семантический поиск v2 с использованием:
    - Графа документов
    - Извлечения сущностей
    - Мультиполевого ранжирования
    - Переранжирования
    """
    if not READY:
        return SearchResponse(query=q, took_ms=0, total_candidates=0, results=[])

    start = time.time()

    # Используем универсальный поисковик если доступен
    if UNIVERSAL_SEARCHER:
        try:
            # Настраиваем параметры поиска
            UNIVERSAL_SEARCHER.cfg.use_graph_search = use_graph
            UNIVERSAL_SEARCHER.cfg.use_entity_extraction = use_entities
            UNIVERSAL_SEARCHER.cfg.top_k_docs = top_k

            # Выполняем поиск
            response = UNIVERSAL_SEARCHER.search(q)

            # Фильтруем по категориям если нужно
            if categories:
                cat_set = set(c.strip() for c in categories.split(","))
                response.results = [
                    r for r in response.results
                    if r.category in cat_set or not cat_set
                ]

            response.took_ms = int((time.time() - start) * 1000)
            return response

        except Exception as e:
            print(f"Ошибка в универсальном поиске: {e}")
            # Fallback на базовый поиск

    # Fallback на базовый поиск v1
    return search_v1(q, top_k, categories=categories)


@app.post("/api/extract_entities")
def extract_entities(query: str):
    """
    Извлечение сущностей из поискового запроса
    """
    if not ENTITY_EXTRACTOR:
        return {"error": "Entity extractor not initialized"}

    try:
        entities = ENTITY_EXTRACTOR.extract(query)
        return entities.to_dict()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/graph/node/{node_id}")
def get_graph_node(node_id: str):
    """
    Получение информации об узле графа
    """
    if not GRAPH_SEARCHER:
        return {"error": "Graph searcher not initialized"}

    try:
        if not GRAPH_SEARCHER.graph_builder.graph.has_node(node_id):
            return {"error": "Node not found"}

        node_data = GRAPH_SEARCHER.graph_builder.graph.nodes[node_id]
        related = GRAPH_SEARCHER.graph_builder.get_related_nodes(
            node_id, max_distance=2)

        return {
            "node_id": node_id,
            "data": dict(node_data),
            "related_nodes": [{"id": nid, "distance": dist} for nid, dist in related[:10]]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/graph/similar/{doc_id}")
def find_similar_documents(doc_id: str, top_k: int = 5):
    """
    Поиск похожих документов через граф
    """
    if not GRAPH_SEARCHER:
        return {"error": "Graph searcher not initialized"}

    try:
        results = GRAPH_SEARCHER.find_similar_documents(doc_id, top_k)
        return {
            "source_doc": doc_id,
            "similar": [r.to_dict() for r in results]
        }
    except Exception as e:
        return {"error": str(e)}
