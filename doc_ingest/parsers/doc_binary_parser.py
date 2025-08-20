# parsers/doc_binary_parser.py
from typing import List, Optional
from pathlib import Path
import os
import subprocess
import shutil
import tempfile

from models import DocElement
from .docx_parser import parse_docx

# ---------- helpers ----------

def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def _which_env_or_system(env_name: str, candidates: list[str]) -> Optional[Path]:
    # 1) env var explicit path
    p = os.environ.get(env_name)
    if p:
        pth = Path(p)
        if pth.exists():
            return pth
    # 2) system PATH
    for c in candidates:
        w = shutil.which(c)
        if w:
            return Path(w)
    return None

def _find_soffice() -> Optional[Path]:
    """
    Find LibreOffice soffice binary across OS:
    - env LO_PATH
    - PATH: soffice / lowriter
    - macOS app bundle
    - Windows typical installs
    """
    p = _which_env_or_system("LO_PATH", ["soffice", "lowriter", "soffice.bin"])
    if p:
        return p

    # macOS app bundle
    mac_paths = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice.bin",
        "/Applications/LibreOffice.app/Contents/MacOS/lowriter",
    ]
    p = _first_existing(mac_paths)
    if p:
        return p

    # Windows typical
    win_paths = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    p = _first_existing(win_paths)
    return p

def _find_antiword() -> Optional[Path]:
    return _which_env_or_system("ANTIWORD_PATH", ["antiword"])

def _find_catdoc() -> Optional[Path]:
    return _which_env_or_system("CATDOC_PATH", ["catdoc"])

def _find_wvtext() -> Optional[Path]:
    # wvText из пакета 'wv'
    return _which_env_or_system("WVTEXT_PATH", ["wvText", "wvtext"])

def _decode_best(data: bytes) -> str:
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")

# ---------- converters ----------

def _lo_convert(path: str, out_dir: Path, fmt: str) -> Optional[Path]:
    """
    LibreOffice headless conversion:
      fmt='docx' -> .docx
      fmt='txt:Text' -> .txt
      fmt='html:XHTML Writer File' -> .html  (опционально)
    """
    soffice = _find_soffice()
    if not soffice:
        return None
    try:
        args = [soffice.as_posix(), "--headless", "--convert-to", fmt, "--outdir", out_dir.as_posix(), path]
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ext = ".docx" if fmt == "docx" else (".txt" if fmt.startswith("txt") else ".html")
        out = out_dir / (Path(path).stem + ext)
        return out if out.exists() else None
    except Exception:
        return None

def _antiword_text(path: str) -> Optional[str]:
    exe = _find_antiword()
    if not exe:
        return None
    try:
        res = subprocess.run([exe.as_posix(), path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return _decode_best(res.stdout)
    except Exception:
        return None

def _catdoc_text(path: str) -> Optional[str]:
    exe = _find_catdoc()
    if not exe:
        return None
    try:
        res = subprocess.run([exe.as_posix(), path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return _decode_best(res.stdout)
    except Exception:
        return None

def _wvtext_text(path: str) -> Optional[str]:
    exe = _find_wvtext()
    if not exe:
        return None
    try:
        # wvText input.doc -  (stdout)
        res = subprocess.run([exe.as_posix(), path, "-"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return _decode_best(res.stdout)
    except Exception:
        return None

# ---------- main parser ----------

def parse_doc_binary(path: str, category: Optional[str] = None) -> List[DocElement]:
    """
    Стратегия:
    1) LibreOffice: doc->txt (быстро) или doc->docx + parse_docx (сохранение структуры таблиц/абзацев лучше).
    2) antiword / catdoc / wvText (CLI) — fallback.
    3) Иначе — предупреждение и graceful skip.
    """
    p = Path(path)
    out: List[DocElement] = []

    # 1) LibreOffice конверт в docx → используем существующий docx-парсер
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # сначала в docx (лучше сохраняет структуру, потом docx_parser вытянет параграфы/таблицы/OMML)
        docx_path = _lo_convert(path, tmp, fmt="docx")
        if docx_path:
            try:
                return parse_docx(docx_path.as_posix(), category=category)
            except Exception as e:
                # если по какой-то причине docx разбор не удался — попробуем в текст
                pass

        # затем попытка в plain text через LO (не так богато, но стабильно)
        txt_path = _lo_convert(path, tmp, fmt="txt:Text")
        if txt_path and txt_path.exists():
            try:
                text = txt_path.read_text(encoding="utf-8", errors="replace")
                order = 0
                for para in [t.strip() for t in text.splitlines()]:
                    if not para:
                        continue
                    out.append(DocElement(
                        category=category, doc_path=path, doc_id=p.stem, source_type="doc",
                        element_type="paragraph", text=para, order=order
                    ))
                    order += 1
                if out:
                    return out
            except Exception:
                pass

    # 2) CLI fallback: antiword / catdoc / wvText
    text = _antiword_text(path) or _catdoc_text(path) or _wvtext_text(path)
    if text:
        order = 0
        for para in [t.strip() for t in text.splitlines()]:
            if not para:
                continue
            out.append(DocElement(
                category=category, doc_path=path, doc_id=p.stem, source_type="doc",
                element_type="paragraph", text=para, order=order
            ))
            order += 1
        return out

    # 3) ничего не вышло — предупреждаем один раз
    print(f"WARNING: .doc not parsed (LibreOffice/antiword/catdoc/wvText not found or failed): {path}")
    return out