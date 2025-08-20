# models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
from uuid import uuid4
from pathlib import Path


ElementType = Literal["paragraph","title","list","table","formula","image","metadata","slide","row"]

class DocElement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    category: Optional[str] = None
    doc_path: Optional[str] = None
    doc_id: Optional[str] = None
    source_type: Optional[str] = None  # pdf/docx/pptx/xlsx/csv/txt/pg
    page: Optional[int] = None
    order: Optional[int] = None
    bbox: Optional[List[float]] = None   # [x0,y0,x1,y1] для PDF
    element_type: ElementType
    text: Optional[str] = None           # основной текст
    html: Optional[str] = None           # табличный/формульный HTML
    media_path: Optional[str] = None     # сохранённое изображение/формула (png)
    headings_path: Optional[List[str]] = None  # ["Гл.1","1.1",...]
    parents: Optional[List[str]] = None        # id узлов-родителей в графе
    metadata: Dict = Field(default_factory=dict)



class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    category: Optional[str] = None
    doc_id: Optional[str] = None
    doc_path: Optional[str] = None
    text: str
    token_len: int
    char_len: int
    chunk_index: int
    from_order: int
    to_order: int
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    heading_path: Optional[List[str]] = None
    element_types: Optional[List[str]] = None
    media_refs: Optional[List[str]] = None