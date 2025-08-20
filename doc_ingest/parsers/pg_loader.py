# parsers/pg_loader.py
from typing import List
from models import DocElement
import psycopg2

def load_from_pg(dsn: str, query: str, category: str="DB") -> List[DocElement]:
    con = psycopg2.connect(dsn)
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    out = []
    order = 0
    for r in rows:
        # адаптируйте под свои колонки
        # пример: (id, title, content)
        if len(r) == 3:
            rid, title, content = r
            text = f"{title}\n{content}"
        else:
            text = " | ".join(str(x) for x in r)
        out.append(DocElement(
            category=category, doc_path="postgres", doc_id=str(r[0]),
            source_type="pg", element_type="paragraph", text=text, order=order
        ))
        order += 1
    cur.close(); con.close()
    return out