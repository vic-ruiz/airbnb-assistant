import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "data/faiss.index"
DB_PATH = "data/kb.sqlite"

class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMB_MODEL)
        self.index = faiss.read_index(INDEX_PATH)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        try:
            self.conn.close()
        except:
            pass

    def retrieve(self, query, k=6, property_id=None):
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q, k)
        ids = [int(i) for i in idxs[0] if i != -1]

        if not ids:
            return []

        placeholders = ",".join(["?"] * len(ids))
        rows = self.conn.execute(
            f"SELECT rowid as rid, * FROM kb WHERE rowid IN ({placeholders})",
            ids
        ).fetchall()

        # Si se indicó property_id, filtramos por propiedad y reordenamos conservando el puntaje aproximado
        if property_id:
            rows = [r for r in rows if r["property_id"] == property_id]

        # Empaquetar con score FAISS
        rid2score = {rid: sc for rid, sc in zip(idxs[0].tolist(), scores[0].tolist()) if rid != -1}
        results = []
        for r in rows:
            results.append({
                "rid": r["rid"],
                "text": r["text"],
                "property_id": r["property_id"],
                "section": r["section"],
                "lang": r["lang"],
                "score": float(rid2score.get(r["rid"], 0.0))
            })

        # ordenar por score descendente
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
