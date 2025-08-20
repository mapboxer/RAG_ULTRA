# parsers/excel_csv_parser.py
from typing import List, Optional
from pathlib import Path
import pandas as pd
import csv

from models import DocElement
from utils_encoding import detect_encoding


def _detect_encoding(path: str) -> str:
    # быстрые попытки
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(4096)
            return enc
        except Exception:
            continue
    # глубокая детекция (опционально)
    try:
        from charset_normalizer import from_path
        res = from_path(path).best()
        if res:
            return res.encoding or "utf-8"
    except Exception:
        pass
    return "utf-8"

def parse_csv(path: str, category: Optional[str]=None) -> List[DocElement]:
    out: List[DocElement] = []
    doc_id = Path(path).stem
    order = 0
    enc = detect_encoding(path)
    with open(path, "r", encoding=enc, errors="replace", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        except Exception:
            dialect = csv.excel
            dialect.delimiter = ";"
        reader = csv.reader(f, dialect)
        headers = next(reader, None)
        for row in reader:
            if headers and len(headers) == len(row):
                text = "; ".join(f"{h}: {v}" for h, v in zip(headers, row))
            else:
                text = "; ".join(row)
            out.append(DocElement(
                category=category, doc_path=path, doc_id=doc_id, source_type="csv",
                element_type="row", text=text, order=order
            ))
            order += 1
    return out

def parse_excel(path: str, category: Optional[str]=None) -> List[DocElement]:
    import warnings
    warnings.filterwarnings("ignore", message="Data Validation extension is not supported")
    out: List[DocElement] = []
    xls = pd.ExcelFile(path)
    doc_id = Path(path).stem
    order = 0
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet)
        for _, row in df.iterrows():
            text = "; ".join(f"{col}: {row[col]}" for col in df.columns)
            out.append(DocElement(
                category=category, doc_path=path, doc_id=doc_id, source_type="xlsx",
                element_type="row", text=str(text), order=order,
                metadata={"sheet": sheet}
            ))
            order += 1
    return out