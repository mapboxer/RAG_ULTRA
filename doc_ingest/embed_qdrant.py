# embed_qdrant.py
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from models import Chunk
from config import EmbeddingConfig
from sentence_transformers import SentenceTransformer

def push_to_qdrant(chunks: List[Chunk], cfg: EmbeddingConfig,
                   model: SentenceTransformer, prefix: str = ""):
    # быстрый фолбэк: если явно не просили qdrant, а он недоступен — просто скипаем
    qdrant_required = getattr(cfg, "qdrant_required", False)

    try:
        client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port, timeout=60.0)
        # ping
        _ = client.get_collections()
    except Exception as e:
        msg = f"WARNING: Qdrant unreachable ({cfg.qdrant_host}:{cfg.qdrant_port}): {e}. Skipping Qdrant push."
        if qdrant_required:
            raise RuntimeError(msg) from e
        print(msg)
        return

    try:
        dim = model.get_sentence_embedding_dimension()
    except Exception:
        dim = len(model.encode([prefix + chunks[0].text], convert_to_numpy=True)[0])

    # создаём коллекцию при необходимости
    existing = [c.name for c in client.get_collections().collections]
    if cfg.qdrant_collection not in existing:
        client.recreate_collection(
            collection_name=cfg.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    # батч-аплоад
    B = 256
    for i in range(0, len(chunks), B):
        batch = chunks[i:i+B]
        texts = [prefix + c.text for c in batch]
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                            batch_size=128, show_progress_bar=False)
        points = []
        for c, v in zip(batch, vecs):
            payload = {
                "id": c.id,
                "category": c.category,
                "doc_id": c.doc_id,
                "doc_path": c.doc_path,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "token_len": c.token_len,
                "heading_path": c.heading_path,
                "page_from": c.page_from,
                "page_to": c.page_to,
                "element_types": c.element_types,
                "media_refs": c.media_refs
            }
            points.append(PointStruct(id=c.id, vector=v.tolist(), payload=payload))
        client.upsert(collection_name=cfg.qdrant_collection, points=points)

    print(f"Qdrant upsert OK -> collection '{cfg.qdrant_collection}'")