import json, sqlite3, os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "data/faiss.index"
DB_PATH = "data/kb.sqlite"
KB_JSONL = "data/kb.jsonl"

def chunk_text(txt, max_chars=900):
    txt = " ".join(txt.split())
    if len(txt) <= max_chars:
        return [txt]
    chunks = []
    start = 0
    while start < len(txt):
        end = min(start + max_chars, len(txt))
        # cortar por espacio para no partir palabras
        if end < len(txt):
            while end > start and txt[end-1] != " ":
                end -= 1
        chunks.append(txt[start:end].strip())
        start = end
    return [c for c in chunks if c]

def build_index():
    assert os.path.exists(KB_JSONL), f"No existe {KB_JSONL}"
    os.makedirs("data", exist_ok=True)

    model = SentenceTransformer(EMB_MODEL)

    texts, meta = [], []
    with open(KB_JSONL, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue  # salta líneas vacías o comentarios
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"KB JSONL inválido en línea {i}: {e.msg} (col {e.colno}). "
                    f"Muestra: {line[:120]}"
                ) from e

            for ch in chunk_text(r["text"], max_chars=900):
                texts.append(ch)
                meta.append({
                    "property_id": r["property_id"],
                    "section": r.get("section","general"),
                    "lang": r.get("lang","es")
                })

    print(f"[KB] Chunks totales: {len(texts)}")
    X = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    X = X.astype("float32")

    # FAISS (cosine via inner product con embeddings normalizados)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, INDEX_PATH)
    print(f"[FAISS] Guardado en {INDEX_PATH}")

    # Persistir metadata + textos en SQLite
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS kb (
        id INTEGER PRIMARY KEY,
        text TEXT,
        property_id TEXT,
        section TEXT,
        lang TEXT
    )""")
    c.execute("DELETE FROM kb")  # rebuild completo
    c.executemany("INSERT INTO kb(text, property_id, section, lang) VALUES (?,?,?,?)",
                  [(t, m["property_id"], m["section"], m["lang"]) for t, m in zip(texts, meta)])
    conn.commit()
    conn.close()
    print(f"[SQLite] Guardado en {DB_PATH}")

if __name__ == "__main__":
    build_index()
