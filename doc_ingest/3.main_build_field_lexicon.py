# tools/build_field_lexicon.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Set, List

IN_META = "outputs/index/doc_fields_meta.json"
OUT_LEX = "outputs/index/lexicon_doc_fields.json"

def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("ё", "е").split())

def _tok_ru(s: str) -> List[str]:
    import re
    return [t for t in re.split(r"[^a-zA-Zа-яА-Я0-9]+", _norm(s)) if t]

def run():
    if not Path(IN_META).exists():
        raise FileNotFoundError(f"{IN_META} not found. Run build_doc_fields_index first.")
    J = json.loads(Path(IN_META).read_text(encoding="utf-8"))
    meta = J["meta"]

    N = len(meta)
    df: Dict[str, int] = {}

    for m in meta:
        # берём токены только из path_text и heading_text — это «навигационные» поля
        toks = set(_tok_ru((m.get("path_text") or "") + " " + (m.get("heading_text") or "")))
        for t in toks:
            df[t] = df.get(t, 0) + 1

    out = {"N": N, "df": df}
    Path(OUT_LEX).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: saved lexicon ({len(df)} tokens) -> {OUT_LEX}")

if __name__ == "__main__":
    run()