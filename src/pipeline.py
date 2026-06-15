"""Orquestador end-to-end del sistema.

Dos modos de ejecución:

- **Baseline (use_rag=False)**: solo LLM, sin contexto.
  Mail crudo → analyze_mail() → MailAnalysis con suggested_response=null.

- **RAG (use_rag=True)**: pipeline completo.
  Mail crudo → retriever.search() → analyze_mail(con contexto) → MailAnalysis
  con suggested_response y sources poblados.

El punto de entrada único `process_mail()` alterna entre ambos modos.
"""
from __future__ import annotations

from typing import Optional

from .llm import analyze_mail
from .schemas import MailAnalysis


DEFAULT_TOP_K = 3


def process_mail(
    raw_mail: str,
    use_rag: bool = False,
    model_tier: str = "fast",
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = False,
) -> MailAnalysis:
    """Procesa un mail entero a través del pipeline correspondiente.

    Args:
        raw_mail: Texto crudo del mail entrante.
        use_rag: Si True, recupera contexto del KB y genera suggested_response.
            Si False, corre el baseline puro sin RAG.
        model_tier: "fast" o "quality" — qué modelo de Gemini usar.
        top_k: Cuántos documentos recuperar del KB cuando use_rag=True.
        verbose: Imprime latencia y tokens.

    Returns:
        MailAnalysis con el resultado del pipeline.
    """
    if not use_rag:
        return analyze_mail(raw_mail, model_tier=model_tier, verbose=verbose)

    # ── Modo RAG ──
    # Import lazy: el retriever carga sentence-transformers + Chroma, que
    # tardan en arrancar. No queremos pagar ese costo si solo usamos baseline.
    from .retriever import search

    retrieved = search(raw_mail, top_k=top_k)

    if verbose:
        print(f"  Retrieved {len(retrieved)} docs from KB:")
        for r in retrieved:
            print(f"    - {r.doc_id} ({r.category}) score={r.score:.4f}")

    # Pasamos los docs como dicts para que el LLM pueda citar por doc_id.
    context = [
        {"doc_id": r.doc_id, "text": r.text, "topic": r.topic}
        for r in retrieved
    ]

    result = analyze_mail(
        raw_mail,
        model_tier=model_tier,
        retrieved_context=context,
        verbose=verbose,
    )

    # Garantizamos que `sources` esté poblado con los doc_id recuperados que
    # el LLM efectivamente citó. Si el LLM no devolvió sources (algunos casos),
    # los completamos con los doc_id recuperados como fallback transparente.
    if not result.sources:
        result.sources = [r.doc_id for r in retrieved]

    return result
