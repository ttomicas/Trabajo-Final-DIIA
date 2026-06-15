"""Búsqueda híbrida sobre el KB (BM25 léxico + vector semántico + RRF).

Implementa la estrategia recomendada por el patrón **Hybrid Retrieval con
Reciprocal Rank Fusion (RRF)**:

1. Búsqueda léxica con BM25 → top-K candidatos por match de tokens.
2. Búsqueda semántica con Chroma + multilingual-e5-large → top-K
   candidatos por similitud coseno.
3. Fusión con RRF: para cada documento, score = Σ 1/(k+rank_en_cada_lista).
   Esto premia documentos que aparecen alto en ambos rankings sin necesidad
   de calibrar pesos entre las dos métricas (que están en escalas distintas).

Decisión documentada para el informe (Sección 2 — Justificación):
- BM25 solo: rinde bien con vocabulario exacto pero falla con paráfrasis e
  idiomas distintos.
- Vector solo: capta semántica pero a veces ignora términos específicos
  (números de pedido, nombres de producto).
- Híbrido + RRF: combina lo mejor de ambos sin tunear pesos manualmente.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .kb_ingest import CHROMA_COLLECTION_NAME, _tokenize_for_bm25


# Parámetro estándar de RRF, según el paper original de Cormack et al. (2009).
# Valores típicos: 60. Subir reduce el impacto del rank, bajar lo aumenta.
RRF_K = 60


@dataclass
class RetrievalResult:
    """Un documento recuperado con su score combinado y rankings de origen."""
    doc_id: str
    text: str
    score: float            # Score RRF combinado
    source_file: str
    category: str
    topic: str
    language: str
    bm25_rank: Optional[int]    # Posición en el ranking BM25 (None si no apareció)
    vector_rank: Optional[int]  # Posición en el ranking vectorial


# ────────────────────────────────────────────────────────────────────────
# Carga (singleton) de los índices BM25 y Chroma
# ────────────────────────────────────────────────────────────────────────

_BM25_CACHE: Optional[Dict] = None
_CHROMA_COLLECTION = None
_ENCODER = None


def _load_bm25(bm25_path: Path) -> Dict:
    global _BM25_CACHE
    if _BM25_CACHE is None:
        with open(bm25_path, "rb") as f:
            _BM25_CACHE = pickle.load(f)
    return _BM25_CACHE


def _load_chroma(chroma_persist_dir: Path):
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=str(chroma_persist_dir))
        _CHROMA_COLLECTION = client.get_collection(CHROMA_COLLECTION_NAME)
    return _CHROMA_COLLECTION


def _load_encoder(model_name: str):
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _ENCODER = SentenceTransformer(model_name)
    return _ENCODER


# ────────────────────────────────────────────────────────────────────────
# Búsquedas individuales
# ────────────────────────────────────────────────────────────────────────

def _bm25_search(query: str, top_k: int, bm25_data: Dict) -> List[Tuple[str, float]]:
    """Devuelve [(doc_id, score), ...] del ranking BM25."""
    bm25 = bm25_data["bm25"]
    chunks = bm25_data["chunks"]
    tokens = _tokenize_for_bm25(query)
    scores = bm25.get_scores(tokens)
    # Ordenamos descendente
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(chunks[i]["doc_id"], float(scores[i])) for i in ranked_idx[:top_k]]


def _vector_search(query: str, top_k: int, collection, encoder
                   ) -> List[Tuple[str, float]]:
    """Devuelve [(doc_id, score), ...] del ranking vectorial."""
    q_emb = encoder.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k)
    ids = res["ids"][0]
    distances = res["distances"][0]
    # Convertimos distancia coseno a similitud (1 - dist) para reportar score.
    return [(doc_id, 1.0 - dist) for doc_id, dist in zip(ids, distances)]


# ────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ────────────────────────────────────────────────────────────────────────

def _rrf_combine(
    bm25_results: List[Tuple[str, float]],
    vector_results: List[Tuple[str, float]],
    k: int = RRF_K,
) -> List[Tuple[str, float, Optional[int], Optional[int]]]:
    """Combina ambos rankings con RRF. Devuelve [(doc_id, score, bm25_rank, vec_rank)]."""
    rrf_scores: Dict[str, float] = {}
    bm25_ranks: Dict[str, int] = {}
    vector_ranks: Dict[str, int] = {}

    for rank, (doc_id, _) in enumerate(bm25_results, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        bm25_ranks[doc_id] = rank

    for rank, (doc_id, _) in enumerate(vector_results, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        vector_ranks[doc_id] = rank

    combined = [
        (doc_id, score,
         bm25_ranks.get(doc_id), vector_ranks.get(doc_id))
        for doc_id, score in rrf_scores.items()
    ]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined


# ────────────────────────────────────────────────────────────────────────
# API pública
# ────────────────────────────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = 5,
    candidates_per_method: int = 10,
    chroma_persist_dir: str = "data/chroma",
    bm25_path: str = "data/bm25.pkl",
    embedding_model: str = "intfloat/multilingual-e5-large",
) -> List[RetrievalResult]:
    """Búsqueda híbrida sobre el KB. Devuelve los top-K documentos relevantes.

    Args:
        query: Texto de la consulta (mail, pregunta, etc.).
        top_k: Cuántos resultados devolver.
        candidates_per_method: Cuántos candidatos pedir a cada método antes de fusionar.
        chroma_persist_dir: Ruta del vector store Chroma.
        bm25_path: Ruta del pickle con el índice BM25.
        embedding_model: Modelo para encoder de queries.
    """
    bm25_data = _load_bm25(Path(bm25_path))
    collection = _load_chroma(Path(chroma_persist_dir))
    encoder = _load_encoder(embedding_model)

    bm25_results = _bm25_search(query, candidates_per_method, bm25_data)
    vector_results = _vector_search(query, candidates_per_method, collection, encoder)
    combined = _rrf_combine(bm25_results, vector_results)

    # Mapeamos doc_id → metadata desde los chunks del BM25 (que tiene todo).
    chunk_index = {c["doc_id"]: c for c in bm25_data["chunks"]}
    results: List[RetrievalResult] = []
    for doc_id, score, bm25_rank, vec_rank in combined[:top_k]:
        c = chunk_index[doc_id]
        results.append(RetrievalResult(
            doc_id=doc_id,
            text=c["text"],
            score=score,
            source_file=c["source_file"],
            category=c["category"],
            topic=c["topic"],
            language=c["language"],
            bm25_rank=bm25_rank,
            vector_rank=vec_rank,
        ))
    return results


def reset_caches() -> None:
    """Borra los singletons. Útil después de re-indexar o cambiar de modelo."""
    global _BM25_CACHE, _CHROMA_COLLECTION, _ENCODER
    _BM25_CACHE = None
    _CHROMA_COLLECTION = None
    _ENCODER = None
