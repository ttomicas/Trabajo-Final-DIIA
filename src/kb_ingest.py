"""Ingesta del knowledge base al vector store + índice léxico.

Lee los archivos Markdown de `data/kb/`, parsea el frontmatter, chunkea el
contenido si es necesario (las FAQs son cortas, normalmente 1 chunk = 1 doc)
y lo indexa por duplicado:

1. **Chroma** con embeddings `multilingual-e5-large` para búsqueda semántica.
2. **BM25** en memoria (persistido como pickle) para búsqueda léxica.

El doble índice habilita la búsqueda híbrida (RRF) que implementa
`src/retriever.py`.

Decisiones de diseño documentadas para el informe:
- Chunk size: 300 tokens (~225 palabras). Las FAQs son cortas, casi siempre
  cabe entera en un chunk. Si superan ~300 tokens, partimos por párrafos con
  overlap 50 tokens para preservar contexto.
- Modelo de embeddings: multilingual-e5-large (1024 dim, 100+ idiomas).
- Vector store: Chroma local persistente — zero-config para PoC.
"""
from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

CHROMA_COLLECTION_NAME = "shophive_kb"


@dataclass
class KBChunk:
    """Una unidad indexada del KB. Cada FAQ produce 1+ chunks."""
    doc_id: str            # ID único del chunk (ej. account_password-reset_0)
    source_file: str       # Path del .md original
    category: str          # ACCOUNT | ORDER | REFUND | PAYMENT | CONTACT
    topic: str             # Tema human-readable
    language: str          # ISO 639-1 ("en", "es", ...)
    text: str              # Texto del chunk


# ────────────────────────────────────────────────────────────────────────
# Parseo del frontmatter Markdown
# ────────────────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_md(filepath: Path) -> Dict[str, str]:
    """Parsea frontmatter YAML y devuelve dict + body."""
    raw = filepath.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {"body": raw.strip()}

    fm_block = m.group(1)
    body = raw[m.end():].strip()

    meta: Dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    meta["body"] = body
    return meta


# ────────────────────────────────────────────────────────────────────────
# Chunking
# ────────────────────────────────────────────────────────────────────────

def _approx_token_count(text: str) -> int:
    """Aproximación rápida: 1 token ~= 0.75 palabras en inglés."""
    return int(len(text.split()) / 0.75)


def chunk_text(text: str, target_tokens: int = 300,
               overlap_tokens: int = 50) -> List[str]:
    """Chunkea por párrafos respetando el target de tokens.

    Si el texto entero cabe en `target_tokens`, devuelve un solo chunk.
    Si no, agrupa párrafos hasta acercarse al target, con overlap leve
    para no perder contexto entre chunks.
    """
    if _approx_token_count(text) <= target_tokens:
        return [text.strip()]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for para in paragraphs:
        para_tokens = _approx_token_count(para)
        if current_tokens + para_tokens > target_tokens and current:
            chunks.append("\n\n".join(current))
            # overlap: arrancamos el siguiente chunk con el último párrafo del anterior
            last_para = current[-1] if current else ""
            last_tokens = _approx_token_count(last_para)
            current = [last_para] if last_tokens <= overlap_tokens else []
            current_tokens = last_tokens if current else 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ────────────────────────────────────────────────────────────────────────
# Carga de chunks desde data/kb/
# ────────────────────────────────────────────────────────────────────────

def load_chunks_from_kb(kb_dir: Path) -> List[KBChunk]:
    """Lee todos los .md de `kb_dir`, los chunkea y devuelve la lista plana."""
    chunks: List[KBChunk] = []
    for md_file in sorted(kb_dir.glob("*.md")):
        meta = _parse_md(md_file)
        body = meta.get("body", "")
        category = meta.get("category", "UNKNOWN")
        topic = meta.get("topic", md_file.stem)
        language = meta.get("language", "en")

        for i, chunk_text_ in enumerate(chunk_text(body)):
            doc_id = f"{md_file.stem}__{i}"
            chunks.append(KBChunk(
                doc_id=doc_id,
                source_file=str(md_file.relative_to(kb_dir.parent.parent)
                                if kb_dir.parent.parent in md_file.parents
                                else md_file),
                category=category,
                topic=topic,
                language=language,
                text=chunk_text_,
            ))
    return chunks


# ────────────────────────────────────────────────────────────────────────
# Indexado en Chroma (vector) y BM25 (léxico)
# ────────────────────────────────────────────────────────────────────────

def _make_embeddings(texts: List[str], model_name: str,
                     batch_size: int = 16) -> List[List[float]]:
    """Genera embeddings con sentence-transformers (multilingual-e5)."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    encoder = SentenceTransformer(model_name)
    # e5 espera prefijo "passage:" para documentos y "query:" para consultas
    prefixed = [f"passage: {t}" for t in texts]
    embs = encoder.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.tolist()


def ingest_kb(
    kb_dir: str = "data/kb",
    chroma_persist_dir: str = "data/chroma",
    bm25_path: str = "data/bm25.pkl",
    embedding_model: str = "intfloat/multilingual-e5-large",
    verbose: bool = True,
) -> int:
    """Indexa todo el KB en Chroma + BM25.

    Returns:
        Cantidad de chunks indexados.
    """
    import chromadb  # type: ignore
    from rank_bm25 import BM25Okapi  # type: ignore

    kb_path = Path(kb_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"No existe el directorio {kb_path.resolve()}")

    chunks = load_chunks_from_kb(kb_path)
    if not chunks:
        raise RuntimeError(f"No se encontraron .md en {kb_path}")

    if verbose:
        print(f"📚 {len(chunks)} chunks cargados de {kb_path}")

    # --- Embeddings ---
    if verbose:
        print(f"🧮 Generando embeddings con {embedding_model}...")
    texts = [c.text for c in chunks]
    embs = _make_embeddings(texts, embedding_model)

    # --- Indexado Chroma ---
    if verbose:
        print(f"💾 Indexando en Chroma ({chroma_persist_dir})...")
    Path(chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_persist_dir)

    # Si la colección existe, la borramos y la recreamos (idempotente).
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        pass
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c.doc_id for c in chunks],
        embeddings=embs,
        documents=[c.text for c in chunks],
        metadatas=[
            {"source_file": c.source_file, "category": c.category,
             "topic": c.topic, "language": c.language}
            for c in chunks
        ],
    )

    # --- Indexado BM25 ---
    if verbose:
        print(f"📖 Indexando BM25 ({bm25_path})...")
    Path(bm25_path).parent.mkdir(parents=True, exist_ok=True)
    tokenized = [_tokenize_for_bm25(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    # Guardamos el modelo + metadata para que retriever.py pueda recargarlo.
    with open(bm25_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunks": [asdict(c) for c in chunks],
            "tokenized": tokenized,
        }, f)

    if verbose:
        print(f"✅ Indexado completo: {len(chunks)} chunks en Chroma + BM25.")
    return len(chunks)


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenización simple para BM25: minúsculas + split alfanumérico."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]
