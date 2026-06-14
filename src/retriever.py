"""Búsqueda semántica sobre el vector store Chroma.

Wrapper fino sobre Chroma + sentence-transformers (multilingual-e5-large).
Provee `search(query, top_k)` para recuperar los documentos más similares
del KB, que se inyectarán como contexto en el prompt del LLM.

TODO (Fase 3): implementar inicialización del client Chroma, carga del
encoder multilingüe y método search().
"""
from __future__ import annotations

from typing import List, Tuple


def search(query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
    """Devuelve los top-k documentos más similares al query.

    Returns:
        Lista de tuplas (texto, score, metadata).
    """
    raise NotImplementedError("Pendiente — Fase 3.")
