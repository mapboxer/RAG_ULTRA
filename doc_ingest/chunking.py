# chunking.py
# -*- coding: utf-8 -*-
"""
Умный семантический чанкинг:
- heading-aware (учёт заголовков/иерархии папок),
- cohesion-aware (разрывы по падению связности соседних предложений),
- budget-aware (целевой/максимальный размер чанка в токенах),
- tables-aware (таблицы как отдельные чанки с контекстом),
- sentence-overlap (перекрытие предложений между соседними чанками).

Модель эмбеддингов и токенайзер передаются извне (офлайн, локально).
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from razdel import sentenize
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

from models import DocElement, Chunk
from config import EmbeddingConfig


# Эвристика заголовков (нумерация/верхний регистр/ключевые слова)
_HEADING_RE = re.compile(
    r"^((раздел|глава|приложение)\s+\d+[\.\)]?|(\d+(\.\d+){0,3})[\)\.]?)\s+.+",
    flags=re.IGNORECASE
)


def _is_heading(text: str) -> bool:
    t = (text or "").strip()
    if len(t) <= 3:
        return False
    if _HEADING_RE.match(t):
        return True
    # короткая строка В ВЕРХНЕМ РЕГИСТРЕ как заголовок
    if len(t) < 120 and t == t.upper() and re.search(r"[А-ЯA-Z]{3,}", t):
        return True
    return False


def _token_count(tok: Optional[PreTrainedTokenizerBase], text: str) -> int:
    """Подсчёт токенов; если токенайзер не передан — грубая оценка."""
    if tok is None:
        # грубая оценка BPE: ~4 символа на токен
        return max(1, int(len(text) / 4))
    # не добавляем спец-токены — работаем с "голым" текстом
    return len(tok.encode(text, add_special_tokens=False))


def _sentences(text: str) -> List[str]:
    return [s.text.strip() for s in sentenize(text or "") if s.text.strip()]


def _adjacent_sim_drop(embs: np.ndarray) -> List[int]:
    """
    Индексы разрывов между соседними предложениями по косинусу.
    Возвращает список позиций i, где разрыв стоит ПОСЛЕ предложения i-1 (т.е. сегмент [0:i], [i:...]).
    """
    if len(embs) < 3:
        return []
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    sims = (embs[:-1] * embs[1:]).sum(axis=1)  # cos между соседними
    thr = float(np.quantile(sims, 0.25))       # нижний квартиль как порог
    cuts = [i + 1 for i, s in enumerate(sims) if s <= thr]
    return cuts


def _segment_by_cohesion_and_budget(
    sents: List[str],
    sent_token_lens: List[int],
    model: SentenceTransformer,
    prefix: str,
    max_tokens: int,
    use_cohesion: bool = True,
) -> List[Tuple[int, int]]:
    """
    Возвращает список сегментов предложений [(start, end), ...] по падениям связности и бюджету.
    """
    if not sents:
        return []

    # Кандидаты разрывов по связности
    cuts_set = set()
    if use_cohesion and len(sents) > 2:
        s_embs = model.encode(
            [prefix + s for s in sents],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        cuts_set = set(_adjacent_sim_drop(s_embs))

    segments: List[Tuple[int, int]] = []
    start = 0
    while start < len(sents):
        end = start
        cur_tokens = 0
        # Наращиваем сегмент по предложениям
        while end < len(sents):
            # остановка на "плохом" стыке связности (если уже что-то набрано)
            if end in cuts_set and end > start:
                break
            # проверка бюджета
            next_tokens = cur_tokens + sent_token_lens[end]
            if next_tokens > max_tokens and end > start:
                break
            cur_tokens = next_tokens
            end += 1
        # гарантируем прогресс хотя бы на одно предложение
        if end == start:
            end = min(start + 1, len(sents))
        segments.append((start, end))
        start = end

    return segments


def build_chunks(
    elements: List[DocElement],
    cfg: EmbeddingConfig,
    model: SentenceTransformer,
    tokenizer: Optional[PreTrainedTokenizerBase],
    prefix: str = "",
) -> List[Chunk]:
    """
    Основная процедура: собирает буферы по элементам, внутри — секционирует по предложениям
    с учётом связности и бюджета, выдаёт чанки.
    """
    Path("outputs/index").mkdir(parents=True, exist_ok=True)

    chunks: List[Chunk] = []
    cur_buf: List[Tuple[DocElement, str]] = []  # (element, text_part)
    cur_tokens = 0
    chunk_index = 0
    media_refs: List[str] = []

    # Иерархия папок -> стартовый heading_path
    heading_path: List[str] = []
    prefix_hierarchy = []
    if elements and elements[0].category:
        prefix_hierarchy = [p for p in Path(elements[0].category).parts]
    heading_path = prefix_hierarchy[:]

    def flush_buf():
        """Сброс текущего буфера в один или несколько чанков."""
        nonlocal chunks, cur_buf, cur_tokens, chunk_index, media_refs

        if not cur_buf:
            return

        # Собираем предложения и карту принадлежности предложений к элементам
        sents: List[str] = []
        owners: List[DocElement] = []  # каждый индекс предложения -> DocElement
        for el, txt in cur_buf:
            ss = _sentences(txt)
            sents.extend(ss)
            owners.extend([el] * len(ss))

        if not sents:
            cur_buf, cur_tokens, media_refs = [], 0, []
            return

        # Токены по предложению (для бюджета)
        sent_token_lens = [ _token_count(tokenizer, s) for s in sents ]

        # Получаем сегменты предложений (границы по связности и бюджету)
        segments = _segment_by_cohesion_and_budget(
            sents=sents,
            sent_token_lens=sent_token_lens,
            model=model,
            prefix=prefix,
            max_tokens=cfg.chunk_max_tokens,
            use_cohesion=cfg.cohesion_split,
        )

        # Перекрытие предложений между чанками
        overlap = max(0, int(cfg.sentence_overlap or 0))

        for si, (st, en) in enumerate(segments):
            # Метаданные по исходным элементам для сегмента
            seg_owners = owners[st:en]
            orders = [o.order for o in seg_owners if isinstance(o.order, int)]
            pages = [o.page for o in seg_owners if isinstance(o.page, int)]
            element_types = list({o.element_type for o in seg_owners})

            # Текст сегмента (учитываем overlap от предыдущего сегмента)
            seg_sents = sents[st:en]
            if overlap > 0 and si > 0:
                prev_st, prev_en = segments[si - 1]
                pre_sents = sents[max(prev_en - overlap, prev_st):prev_en]
                candidate = "\n".join(pre_sents + seg_sents)
                # если overlap сильно раздувает — сокращаем перекрытие
                while _token_count(tokenizer, candidate) > cfg.chunk_max_tokens and len(pre_sents) > 0:
                    pre_sents = pre_sents[1:]
                    candidate = "\n".join(pre_sents + seg_sents)
                text = candidate
            else:
                text = "\n".join(seg_sents)

            tokens = _token_count(tokenizer, text)

            # Слияние слишком мелких сегментов с предыдущим чанком (если влезает)
            if tokens < cfg.min_chunk_tokens and chunks:
                prev = chunks[-1]
                merged = prev.text + "\n" + text
                if _token_count(tokenizer, merged) <= cfg.chunk_max_tokens:
                    prev.text = merged
                    prev.token_len = _token_count(tokenizer, prev.text)
                    prev.char_len = len(prev.text)
                    if orders:
                        prev.from_order = min(prev.from_order, min(orders))
                        prev.to_order = max(prev.to_order, max(orders))
                    if pages:
                        prev.page_from = min(prev.page_from or pages[0], min(pages))
                        prev.page_to = max(prev.page_to or pages[-1], max(pages))
                    # типы и media_refs объединяем
                    if element_types:
                        prev.element_types = sorted(list(set((prev.element_types or []) + element_types)))
                    if media_refs:
                        prev.media_refs = sorted(list(set((prev.media_refs or []) + media_refs)))
                    continue

            # Нормальный чанк
            cat = cur_buf[0][0].category
            doc_id = cur_buf[0][0].doc_id
            doc_path = cur_buf[0][0].doc_path

            chunks.append(Chunk(
                category=cat,
                doc_id=doc_id,
                doc_path=doc_path,
                text=text,
                token_len=tokens,
                char_len=len(text),
                chunk_index=chunk_index,
                from_order=min(orders) if orders else 0,
                to_order=max(orders) if orders else 0,
                page_from=min(pages) if pages else None,
                page_to=max(pages) if pages else None,
                heading_path=heading_path[:] if heading_path else None,
                element_types=element_types,
                media_refs=media_refs[:] if media_refs else None
            ))
            chunk_index += 1

        # Очистка буфера
        cur_buf, cur_tokens, media_refs = [], 0, []

    # Основной проход по элементам документа
    for el in elements:
        # Заголовки: при обнаружении — сбрасываем буфер и обновляем heading_path
        if cfg.heading_aware and el.element_type in ("paragraph", "title") and el.text:
            if _is_heading(el.text):
                flush_buf()
                # Эвристический уровень заголовка по числовой нотации
                m = re.match(r"^(\d+(?:\.\d+){0,3})", el.text.strip())
                depth = (len(m.group(1).split(".")) if m else 1) - 1
                depth = max(0, depth)
                if len(heading_path) > len(prefix_hierarchy) + depth:
                    heading_path = heading_path[: len(prefix_hierarchy) + depth]
                elif len(heading_path) < len(prefix_hierarchy):
                    heading_path = prefix_hierarchy[:]
                heading_path.append(el.text.strip())
                continue

        # Таблица — отдельный чанк (с контекстом заголовков)
        if el.element_type == "table" and cfg.table_as_is:
            flush_buf()
            tbl_text = el.text or (el.html or "")
            if heading_path:
                tbl_text = f"{' > '.join(heading_path)}\n{tbl_text}"
            # эмитим как отдельный буфер
            cur_buf = [(el, tbl_text)]
            flush_buf()
            continue

        # Изображения/формулы — собираем ссылки (текст берём отдельно, если был)
        if el.element_type in ("image", "formula") and el.media_path:
            media_refs.append(el.media_path)

        # Обычный текстовый элемент
        piece = (el.text or el.html or "").strip()
        if not piece:
            continue

        piece_tokens = _token_count(tokenizer, piece)

        # Контроль целевого бюджета для "текущего" буфера (не максимального!)
        if cur_tokens + piece_tokens > cfg.chunk_target_tokens:
            flush_buf()
        cur_buf.append((el, piece))
        cur_tokens += piece_tokens

    # финальный сброс
    flush_buf()
    return chunks