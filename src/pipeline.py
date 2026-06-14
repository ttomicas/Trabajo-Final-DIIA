"""Orquestador end-to-end del sistema.

Compose los módulos del paquete según la fase activa:

    Fase 1 (no RAG):
        mail crudo  →  analyze_mail()  →  MailAnalysis

    Fase 3 (RAG completo, implementado más adelante):
        mail crudo  →  pii.redact_text()
                    →  retriever.search()
                    →  analyze_mail(con contexto)
                    →  MailAnalysis (con suggested_response y sources)

El punto de entrada único `process_mail()` acepta `use_rag` para alternar
entre el baseline (Fase 1) y el sistema completo (Fase 3+).
"""
from __future__ import annotations

from typing import Optional

from .llm import analyze_mail
from .schemas import MailAnalysis


def process_mail(
    raw_mail: str,
    use_rag: bool = False,
    model_tier: str = "fast",
    verbose: bool = False,
) -> MailAnalysis:
    """Procesa un mail entero a través del pipeline correspondiente.

    Args:
        raw_mail: Texto crudo del mail entrante.
        use_rag: Si True (Fase 3+), recupera contexto del KB. Si False (Fase 1),
            corre el baseline sin RAG.
        model_tier: "fast" o "quality" — qué modelo de Gemini usar.
        verbose: Imprime latencia y tokens.

    Returns:
        MailAnalysis con el resultado del pipeline.
    """
    if use_rag:
        # TODO Fase 3: integrar pii.redact_text() + retriever.search() +
        # construcción de few_shot_examples
        raise NotImplementedError(
            "El modo RAG (use_rag=True) se implementa en la Fase 3 del roadmap."
        )

    # Fase 1: baseline sin RAG. Solo LLM con prompt enriquecido.
    return analyze_mail(raw_mail, model_tier=model_tier, verbose=verbose)
