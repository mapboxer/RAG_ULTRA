# utils_io.py
import json, os
from typing import Iterable
from pathlib import Path
from models import DocElement

def ensure_dirs():
    Path("outputs/dataset").mkdir(parents=True, exist_ok=True)
    Path("outputs/media").mkdir(parents=True, exist_ok=True)
    Path("outputs/graphs").mkdir(parents=True, exist_ok=True)

def dump_jsonl(elements: Iterable[DocElement], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for el in elements:
            f.write(json.dumps(el.model_dump(), ensure_ascii=False) + "\n")