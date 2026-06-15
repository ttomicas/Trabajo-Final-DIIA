"""Métricas de evaluación para el sistema RAG.

Cubre las 3 dimensiones que la rúbrica del informe pide para Track A:

1. **Retrieval** (determinísticas, sin LLM):
   - Precision@k, Recall@k, MRR, nDCG@k

2. **Generación** (con LLM-as-judge):
   - Faithfulness: ¿la respuesta solo usa información del contexto?
   - Answer Relevance: ¿la respuesta efectivamente aborda lo que pide el mail?

3. **Operativas**:
   - Latencia p50/p95, tokens, costo estimado.

Todas las funciones son puras y testeables (no hacen IO salvo las de
LLM-as-judge, que llaman a Gemini).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


# ════════════════════════════════════════════════════════════════════════
# Métricas de retrieval (determinísticas)
# ════════════════════════════════════════════════════════════════════════

def precision_at_k(retrieved_ids: List[str], relevant_ids: Iterable[str],
                   k: int) -> float:
    """De los top-k recuperados, qué fracción son relevantes."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Iterable[str],
                k: int) -> float:
    """De los relevantes que hay, qué fracción aparece en top-k."""
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    return len(top_k_set & relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Iterable[str]) -> float:
    """1 / posición del primer relevante recuperado. 0 si no aparece."""
    relevant_set = set(relevant_ids)
    for i, d in enumerate(retrieved_ids, 1):
        if d in relevant_set:
            return 1.0 / i
    return 0.0


def _dcg(scores: List[float]) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(retrieved_ids: List[str], relevance: Dict[str, float],
              k: int) -> float:
    """Normalized DCG con relevance graduada (0-1).

    Args:
        retrieved_ids: orden de docs recuperados por el sistema.
        relevance: mapa doc_id → score (0..1). Docs no presentes valen 0.
        k: cuántos considerar.
    """
    if k <= 0 or not relevance:
        return 0.0
    actual_scores = [relevance.get(d, 0.0) for d in retrieved_ids[:k]]
    ideal_scores = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_scores)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(actual_scores) / ideal_dcg


# ════════════════════════════════════════════════════════════════════════
# Resultados agregados
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalMetrics:
    n: int
    precision_at_1: float
    precision_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float


def aggregate_retrieval_metrics(per_query: List[Dict[str, float]]
                                ) -> RetrievalMetrics:
    """Promedia las métricas por query."""
    if not per_query:
        return RetrievalMetrics(0, 0, 0, 0, 0, 0)
    n = len(per_query)
    return RetrievalMetrics(
        n=n,
        precision_at_1=sum(q["P@1"]    for q in per_query) / n,
        precision_at_3=sum(q["P@3"]    for q in per_query) / n,
        recall_at_5   =sum(q["R@5"]    for q in per_query) / n,
        mrr           =sum(q["MRR"]    for q in per_query) / n,
        ndcg_at_5     =sum(q["nDCG@5"] for q in per_query) / n,
    )


# ════════════════════════════════════════════════════════════════════════
# LLM-as-judge: Faithfulness y Answer Relevance
# ════════════════════════════════════════════════════════════════════════

FAITHFULNESS_JUDGE_PROMPT = """\
Sos un evaluador estricto de un sistema RAG.

Te paso:
1. El CONTEXTO que el sistema RAG recibió (FAQs del KB).
2. La RESPUESTA SUGERIDA que el sistema generó al cliente.

Tu trabajo es evaluar Faithfulness: ¿la respuesta está fundamentada
exclusivamente en el contexto, o introduce afirmaciones no respaldadas?

REGLAS DE EVALUACIÓN
- Una afirmación fáctica (plazos, montos, URLs, políticas, procedimientos) que
  NO esté en el contexto es una violación de faithfulness.
- Frases genéricas de cortesía o reformulaciones del problema del cliente NO
  cuentan como violaciones.
- Si la respuesta dice "no tengo información para responder", eso es faithful (no inventa).

DEVOLVÉ JSON con dos campos:
- score: número entre 0.0 (alucinó mucho) y 1.0 (totalmente respaldada).
- reason: una frase corta justificando el score, mencionando afirmaciones
  problemáticas si existen.
"""

ANSWER_RELEVANCE_JUDGE_PROMPT = """\
Sos un evaluador estricto de un sistema de atención al cliente.

Te paso:
1. El MAIL del cliente.
2. La RESPUESTA SUGERIDA que el sistema generó.

Tu trabajo es evaluar Answer Relevance: ¿la respuesta efectivamente aborda
lo que el cliente pidió?

REGLAS DE EVALUACIÓN
- 1.0: la respuesta resuelve o orienta claramente sobre el problema.
- 0.7: aborda el tema pero falta especificidad o algún paso.
- 0.4: tangencial al pedido, parcialmente útil.
- 0.0: ignora el pedido o cambia de tema.

DEVOLVÉ JSON con dos campos:
- score: número entre 0.0 y 1.0.
- reason: una frase corta justificando el score.
"""


@dataclass
class JudgeResult:
    score: float
    reason: str


def _judge_call(system_prompt: str, user_payload: str,
                model_tier: str = "fast") -> JudgeResult:
    """Llamada genérica a Gemini como juez. Devuelve score + razón."""
    from pydantic import BaseModel, Field

    from .llm import _MODEL_TIERS, _get_client

    class _JudgeSchema(BaseModel):
        score: float = Field(..., description="Score entre 0.0 y 1.0.")
        reason: str  = Field(..., description="Justificación corta.")

    from google.genai import types  # type: ignore

    client = _get_client()
    model_name = _MODEL_TIERS[model_tier]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=_JudgeSchema,
        temperature=0.0,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=user_payload,
        config=config,
    )
    parsed = response.parsed
    if parsed is None:
        import json as _json
        parsed = _JudgeSchema.model_validate(_json.loads(response.text))
    return JudgeResult(score=float(parsed.score), reason=parsed.reason)


def judge_faithfulness(retrieved_context: List[str], suggested_response: str,
                       model_tier: str = "fast") -> JudgeResult:
    """Evalúa si la respuesta sugerida está fundamentada en el contexto."""
    payload = (
        f"=== CONTEXTO ===\n{chr(10).join(retrieved_context)}\n\n"
        f"=== RESPUESTA SUGERIDA ===\n{suggested_response}"
    )
    return _judge_call(FAITHFULNESS_JUDGE_PROMPT, payload, model_tier)


def judge_answer_relevance(mail: str, suggested_response: str,
                           model_tier: str = "fast") -> JudgeResult:
    """Evalúa si la respuesta efectivamente aborda el mail del cliente."""
    payload = (
        f"=== MAIL ===\n{mail}\n\n"
        f"=== RESPUESTA SUGERIDA ===\n{suggested_response}"
    )
    return _judge_call(ANSWER_RELEVANCE_JUDGE_PROMPT, payload, model_tier)
