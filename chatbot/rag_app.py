from __future__ import annotations

import os
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv  # pip install python-dotenv
load_dotenv()


# =====================
# Config
# =====================
STORAGE_DIR = Path("storage")
INDEX_PATH = STORAGE_DIR / "faiss.index"
META_PATH = STORAGE_DIR / "meta.json"

EMB_MODEL_NAME = "intfloat/multilingual-e5-base"
E5_Q_PREFIX = "query: "
E5_D_PREFIX = "passage: "

# OpenAI key kontrolü
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
USE_OPENAI = bool(OPENAI_API_KEY)

# Chunking
CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP = 200

# =====================
# FastAPI
# =====================
app = FastAPI(title="Nutuk RAG API", version="2.0")

# Globals
_embedder: Optional[SentenceTransformer] = None
_index: Optional[faiss.IndexFlatIP] = None
_meta: List[Dict[str, Any]] = []

# =====================
# Models
# =====================
class IngestIn(BaseModel):
    dir_path: str = Field(..., description="Metin dosyalarının (.txt) bulunduğu klasör yolu")

class IngestOut(BaseModel):
    ok: bool
    added_chunks: int
    total_chunks: int

class ChatIn(BaseModel):
    query: str
    top_k: int = 4

class ChatOut(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]

# =====================
# Utils
# =====================
def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMB_MODEL_NAME)
    return _embedder

def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        cut = end
        slice_ = text[start:end]
        period = slice_.rfind(". ")
        if period != -1 and (end - (start + period)) < 250:
            cut = start + period + 1
        chunk = text[start:cut].strip()
        if len(chunk) > 0:
            chunks.append(chunk)
        start = max(cut - overlap, 0) + overlap
    return [c for c in chunks if len(c) > 10]

def embed_passages(passages: List[str]) -> np.ndarray:
    emb = get_embedder()
    vecs = emb.encode([E5_D_PREFIX + p for p in passages], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vecs.astype("float32")

def embed_query(q: str) -> np.ndarray:
    emb = get_embedder()
    vec = emb.encode([E5_Q_PREFIX + q], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.astype("float32")

def load_index_from_disk() -> bool:
    global _index, _meta
    if INDEX_PATH.exists() and META_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))
        with META_PATH.open("r", encoding="utf-8") as f:
            _meta = json.load(f)
        if _index.d != get_embedder().get_sentence_embedding_dimension():
            raise RuntimeError("FAISS index dimension mismatch.")
        return True
    return False

def save_index_to_disk():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    assert _index is not None
    faiss.write_index(_index, str(INDEX_PATH))
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(_meta, f, ensure_ascii=False, indent=2)

def read_all_text_files(root: Path) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    for p in root.rglob("*.txt"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            txt = txt.strip()
            if txt:
                files.append((str(p), txt))
        except Exception:
            continue
    return files

def ensure_index_initialized(dim: int):
    global _index
    if _index is None:
        _index = faiss.IndexFlatIP(dim)

def add_documents(passages: List[str], metas: List[Dict[str, Any]]) -> int:
    global _index, _meta
    if not passages:
        return 0
    X = embed_passages(passages)
    ensure_index_initialized(X.shape[1])
    _index.add(X)
    _meta.extend(metas)
    save_index_to_disk()
    return len(passages)

# =====================
# Optional: OpenAI summarizer (düzenlenmiş)
# =====================
def summarize_with_openai(question: str, contexts: List[str]) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ortam değişkeni ayarlı değil!")

    client = OpenAI(api_key=api_key)

    prompt = (
        "Aşağıda Nutuk'tan alınmış pasajlar var. "
        "Sadece bu pasajlara dayanarak soruyu kısa, net ve doğru şekilde cevapla. "
        "Uydurma bilgi ekleme. Türkçe yanıt ver.\n\n"
        f"Soru: {question}\n\n"
        "Bağlamlar:\n"
    )
    for i, ctx in enumerate(contexts, start=1):
        prompt += f"[{i}] {ctx}\n\n"

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Sen Nutuk konusunda yardımcı bir asistansın."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


# =====================
# Startup
# =====================
@app.on_event("startup")
def _startup_load():
    try:
        load_index_from_disk()
    except Exception:
        pass

# =====================
# Endpoints
# =====================
@app.post("/ingest", response_model=IngestOut)
def ingest(req: IngestIn) -> IngestOut:
    src_dir = Path(req.dir_path).expanduser().resolve()
    if not src_dir.exists() or not src_dir.is_dir():
        raise HTTPException(400, f"Klasör bulunamadı: {src_dir}")

    files = read_all_text_files(src_dir)
    if not files:
        raise HTTPException(400, "Klasörde .txt dosyası bulunamadı.")

    passages: List[str] = []
    metas: List[Dict[str, Any]] = []
    for src, content in files:
        chunks = chunk_text(content)
        for i, ch in enumerate(chunks):
            passages.append(ch)
            metas.append({
                "id": str(uuid.uuid4()),
                "source": src.replace("\\\\", "\\"),
                "chunk_id": i,
                "chars": len(ch),
                "text": ch
            })

    if not load_index_from_disk():
        ensure_index_initialized(get_embedder().get_sentence_embedding_dimension())

    prev_total = len(_meta)
    added = add_documents(passages, metas)
    total = prev_total + added

    return IngestOut(ok=True, added_chunks=added, total_chunks=total)

@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn) -> ChatOut:
    global _index, _meta
    if _index is None or not _meta:
        raise HTTPException(400, "Önce /ingest ile belge ekleyiniz.")

    top_k = max(1, min(12, req.top_k))

    qv = embed_query(req.query)
    D, I = _index.search(np.expand_dims(qv, 0), top_k)

    contexts: List[str] = []
    citations: List[Dict[str, Any]] = []
    seen = set()
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        m = _meta[idx]
        key = (m.get("source"), m.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        txt = m.get("text", "")
        if txt:
            contexts.append(txt)
        citations.append({
            "source": m.get("source"),
            "chunk_id": m.get("chunk_id"),
            "score": float(score),
        })

    if not contexts:
        return ChatOut(answer="Uygun bağlam bulunamadı.", citations=citations)

    api_key_now = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key_now:
        try:
           print("[RAG] OpenAI özetleyici AKTİF")  # <-- ekle
           answer = summarize_with_openai(req.query, contexts)
        except Exception as e:
           print("OpenAI isteğinde hata oluştu:", e)
           answer = "LLM özetlemesi başarısız oldu, bağlamdan ham pasajlar:\n" + "\n\n".join(contexts[:3])
    else:
        print("[RAG] OpenAI ÖZETLEYİCİ PASİF (anahtar yok)")  # <-- ekle
        answer = "\n".join(contexts[:2])

    return ChatOut(answer=answer, citations=citations)



