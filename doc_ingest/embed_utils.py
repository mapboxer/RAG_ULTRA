# embed_utils.py
import os
from pathlib import Path
from typing import Tuple, Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from config import EmbeddingConfig

def load_embedding_stack(cfg: EmbeddingConfig) -> Tuple[SentenceTransformer, Optional[AutoTokenizer], str]:
    # Оффлайн-режим
    if cfg.offline_mode:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # SBERT модель с диска, без сети
    model = SentenceTransformer(
        cfg.model_name,
        device=None if cfg.device == "auto" else cfg.device
    )

    # токенайзер: сперва пытаемся в подпапке 0_Transformer, затем в корне
    tok = None
    candidates = []
    if cfg.tokenizer_subfolder:
        candidates.append(Path(cfg.model_name) / cfg.tokenizer_subfolder)
    candidates.append(Path(cfg.model_name))
    for cand in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(
                cand.as_posix(),
                local_files_only=cfg.local_files_only
            )
            break
        except Exception:
            continue

    prefix = "passage: " if cfg.use_e5_prefix else ""
    return model, tok, prefix