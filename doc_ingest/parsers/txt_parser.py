# parsers/txt_parser.py
from typing import List, Optional
from pathlib import Path
from models import DocElement
from utils_encoding import detect_encoding

def parse_txt(path: str, category: Optional[str] = None) -> List[DocElement]:
    enc = detect_encoding(path)
    with open(path, "r", encoding=enc, errors="replace") as f:
        text = f.read()
    doc_id = Path(path).stem
    return [DocElement(
        category=category, doc_path=path, doc_id=doc_id, source_type="txt",
        element_type="paragraph", text=text, order=0
    )]