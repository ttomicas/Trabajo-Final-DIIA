"""Ingesta de la base de conocimiento (FAQs, políticas) al vector store.

Lee archivos Markdown de `data/kb/`, los chunkea, genera embeddings con
multilingual-e5-large y los persiste en Chroma.

TODO (Fase 3): implementar ingest_kb() con chunking, embedding y upsert
en Chroma.
"""
from __future__ import annotations

from pathlib import Path


def ingest_kb(kb_dir: Path) -> int:
    """Indexa todos los .md de `kb_dir` en Chroma. Devuelve la cantidad de chunks indexados."""
    raise NotImplementedError("Pendiente — Fase 3.")
