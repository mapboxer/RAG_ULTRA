# utils_encoding.py
from typing import Optional

# Опциональные зависимости: используем, если есть
try:
    from charset_normalizer import from_path as cn_from_path  # type: ignore
except Exception:
    cn_from_path = None  # noqa
try:
    import chardet  # type: ignore
except Exception:
    chardet = None  # noqa


def detect_encoding(path: str, quick_try: bool = True) -> str:
    """
    Детекция кодировки файла с многоступенчатым фолбэком:
    1) быстрые попытки utf-8-sig / utf-8 / cp1251;
    2) charset-normalizer (если установлен);
    3) chardet (если установлен);
    4) возврат 'utf-8' по умолчанию.
    """
    if quick_try:
        for enc in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                with open(path, "r", encoding=enc) as f:
                    f.read(1024)
                return enc
            except Exception:
                continue

    if cn_from_path is not None:
        try:
            res = cn_from_path(path).best()
            if res and res.encoding:
                return res.encoding
        except Exception:
            pass

    if chardet is not None:
        try:
            with open(path, "rb") as f:
                raw = f.read(4096)
            guess = chardet.detect(raw)
            enc = guess.get("encoding")
            if enc:
                return enc
        except Exception:
            pass

    return "utf-8"