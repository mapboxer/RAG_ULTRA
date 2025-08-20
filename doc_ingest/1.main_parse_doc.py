# main_parse_doc.py
import os
import sys
import warnings
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path

from config import PipelineConfig
from models import DocElement
from utils_io import ensure_dirs, dump_jsonl
from graph_builder import build_graph, save_graph

from parsers.pdf_parser import parse_pdf
from parsers.docx_parser import parse_docx
from parsers.pptx_parser import parse_pptx
from parsers.excel_csv_parser import parse_excel, parse_csv
from parsers.txt_parser import parse_txt
from parsers.pg_loader import load_from_pg
from parsers.html_md_parser import parse_html_md
from parsers.doc_binary_parser import parse_doc_binary

warnings.filterwarnings(
    "ignore", message="Data Validation extension is not supported")

SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv", ".txt",
             ".md", ".html", ".htm", ".doc"}


def _skip_path(p: Path) -> bool:
    s = p.as_posix()
    if ".ipynb_checkpoints" in s:
        return True
    if p.name.startswith("~$"):  # временные офисные файлы
        return True
    return False


def route_parse(path: str, cat: str, cfg: PipelineConfig) -> List[DocElement]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path, category=cat,
                         keep_images=cfg.keep_images,
                         table_extraction=cfg.pdf_table_extraction,
                         ocr_fallback=cfg.pdf_ocr_fallback)
    if ext == ".docx":
        return parse_docx(path, category=cat)
    if ext == ".doc":
        return parse_doc_binary(path, category=cat)   # <-- НОВОЕ
    if ext in (".html", ".htm", ".md"):
        return parse_html_md(path, category=cat)      # <-- НОВОЕ
    if ext == ".pptx":
        return parse_pptx(path, category=cat)
    if ext in (".xlsx", ".xls"):
        return parse_excel(path, category=cat)
    if ext == ".csv":
        return parse_csv(path, category=cat)
    if ext == ".txt":
        return parse_txt(path, category=cat)
    return []


def run_pipeline(cfg: PipelineConfig):
    ensure_dirs()
    all_elements: List[DocElement] = []
    per_doc: Dict[str, List[DocElement]] = {}

    # 1) Файлы по категориям
    for cat, files in cfg.categories.items():
        for f in tqdm(files, desc=f"[{cat}]"):
            try:
                p = Path(f)
                if _skip_path(p):
                    continue
                if not p.is_file():
                    # мягко пропускаем, без падения
                    print(f"SKIP (not a file): {f}")
                    continue
                if p.suffix.lower() not in SUPPORTED:
                    print(f"SKIP (unsupported ext): {f}")
                    continue
                elems = route_parse(f, cat, cfg)
                all_elements.extend(elems)
                key = f"{cat}::{f}"
                per_doc[key] = elems
            except Exception as e:
                print(f"ERROR {f}: {e}")

    # 2) Источник из PostgreSQL (опционально)
    if cfg.pg_dsn and cfg.pg_query:
        db_elems = load_from_pg(cfg.pg_dsn, cfg.pg_query, category="DB")
        all_elements.extend(db_elems)
        per_doc["DB::postgres"] = db_elems

    # 3) Сохранение
    dump_jsonl(all_elements, "outputs/dataset/elements.jsonl")
    G = build_graph(per_doc)
    save_graph(G, name="corpus_graph")
    print(
        f"OK: {len(all_elements)} элементов. JSONL: outputs/dataset/elements.jsonl")

# collect all path to files


def collect_all_files(root_path):
    """
    Рекурсивно собирает все пути до файлов из всех папок,
    создавая плоскую структуру: папка -> список файлов в ней

    Args:
        root_path (str): Корневая папка для сканирования

    Returns:
        dict: Словарь где ключ - путь к папке, значение - список файлов в этой папке
    """
    result = {}

    # Проходим по всем элементам в файловой системе начиная с root_path
    for root, dirs, files in os.walk(root_path):
        # Фильтруем системные файлы
        files = [f for f in files if not f.startswith('.')]

        # Если в папке есть файлы, добавляем их в результат
        if files:
            file_paths = []
            for file in files:
                full_path = os.path.join(root, file)
                file_paths.append(full_path)

            # Используем относительный путь папки как ключ
            folder_key = os.path.relpath(root, root_path)
            if folder_key == '.':  # Если это корневая папка
                folder_key = os.path.basename(root_path)

            result[folder_key] = file_paths

    return result


if __name__ == "__main__":

    # Проверяем существование папки doc
    doc_path = Path('doc')
    if not doc_path.exists():
        print(f"ОШИБКА: Папка '{doc_path}' не найдена!")
        print("Создайте папку 'doc' и поместите туда документы для обработки.")
        sys.exit(1)

    categories = collect_all_files('doc')

    if not categories:
        print("ПРЕДУПРЕЖДЕНИЕ: В папке 'doc' не найдено поддерживаемых файлов!")
        print("Поддерживаемые форматы:", SUPPORTED)
        sys.exit(0)

    cfg = PipelineConfig(
        categories=categories,
        pdf_ocr_fallback=True,
        pdf_table_extraction=True,
        keep_images=True,
        pg_dsn=None,  # при необходимости
        pg_query=None
    )
    run_pipeline(cfg)
