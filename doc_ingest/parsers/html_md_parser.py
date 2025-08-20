# parsers/html_md_parser.py
from typing import List, Optional
from pathlib import Path
from bs4 import BeautifulSoup
import shutil, base64
from markdown import markdown
from models import DocElement
from utils_encoding import detect_encoding


def _detect_encoding(path: str) -> str:
    try:
        res = from_path(path).best()
        if res and res.encoding:
            return res.encoding
    except Exception:
        pass
    # быстрые варианты
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(1024)
            return enc
        except Exception:
            continue
    return "utf-8"

def _read_md_to_html(path: str) -> str:
    enc = detect_encoding(path)
    with open(path, "r", encoding=enc, errors="replace") as f:
        md_text = f.read()
    return markdown(md_text, extensions=["tables", "fenced_code", "footnotes", "toc"])

def _read_html_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _save_img(src: str, base_dir: Path, stem: str, idx: int) -> Optional[str]:
    media_dir = Path("outputs/media"); media_dir.mkdir(parents=True, exist_ok=True)
    # data URI
    if src.startswith("data:"):
        try:
            header, b64 = src.split(",", 1)
            data = base64.b64decode(b64)
            out = media_dir / f"{stem}_img_{idx}.png"
            with open(out, "wb") as w:
                w.write(data)
            return out.as_posix()
        except Exception:
            return None
    # локальный путь/URL
    p = (base_dir / src).resolve()
    if p.exists() and p.is_file():
        out = media_dir / f"{stem}_img_{idx}{p.suffix.lower() or '.png'}"
        try:
            shutil.copy(p.as_posix(), out.as_posix())
            return out.as_posix()
        except Exception:
            return None
    return None

def parse_html_md(path: str, category: Optional[str]=None) -> List[DocElement]:
    p = Path(path)
    stem = p.stem
    base_dir = p.parent

    # получаем HTML
    if p.suffix.lower() == ".md":
        html = _read_md_to_html(path)
        soup = BeautifulSoup(html, "lxml")
    else:
        data = _read_html_bytes(path)
        soup = BeautifulSoup(data, "lxml")  # bs4 сам определит encoding/meta

    out: List[DocElement] = []
    doc_id = stem
    order = 0

    # обход в приближённом порядке документа
    body = soup.body or soup
    if not body:
        return out

    def emit_text(t: str, etype: str, meta: Optional[dict]=None):
        nonlocal order
        t = (t or "").strip()
        if not t:
            return
        out.append(DocElement(
            category=category, doc_path=path, doc_id=doc_id, source_type=p.suffix.lower().lstrip("."),
            element_type=etype, text=t, order=order,
            metadata=(meta or {})
        ))
        order += 1

    # заголовки, параграфы, списки, таблицы, коды, изображения
    img_idx = 0
    for el in body.descendants:
        if not getattr(el, "name", None):
            continue
        name = el.name.lower()

        if name in ("h1","h2","h3","h4","h5","h6"):
            emit_text(el.get_text(" ", strip=True), "title", {"tag": name})
        elif name == "p":
            emit_text(el.get_text(" ", strip=True), "paragraph")
        elif name in ("li",):
            emit_text(el.get_text(" ", strip=True), "list")
        elif name in ("pre", "code"):
            emit_text(el.get_text("\n", strip=True), "paragraph", {"code": True, "tag": name})
        elif name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td","th"])]
                rows.append("\t".join(cells))
            if rows:
                emit_text("\n".join(rows), "table")
        elif name == "img":
            src = el.get("src")
            if not src:
                continue
            mpath = _save_img(src, base_dir, stem, img_idx)
            img_idx += 1
            if mpath:
                out.append(DocElement(
                    category=category, doc_path=path, doc_id=doc_id, source_type=p.suffix.lower().lstrip("."),
                    element_type="image", media_path=mpath, order=order,
                    metadata={"alt": el.get("alt")}
                ))
                order += 1

    return out