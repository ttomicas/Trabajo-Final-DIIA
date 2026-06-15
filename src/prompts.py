"""Plantillas de prompts para Gemini (modos baseline y RAG).

Dos modos:

- `SYSTEM_PROMPT_BASELINE`: el LLM analiza el mail sin contexto del KB. Devuelve
  intent + summary + urgency + language + entities, pero deja `suggested_response`
  en null. Es el modo "naive baseline" del Track A.

- `SYSTEM_PROMPT_RAG`: el LLM recibe documentos relevantes recuperados del KB
  y genera además `suggested_response` basada *exclusivamente* en ese contexto.
  Lista en `sources` los doc_ids efectivamente usados. Este es el sistema final.

`build_user_prompt` arma el mensaje de usuario combinando mail + contexto (si lo
hay) + few-shot examples (opcional).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Union


# ────────────────────────────────────────────────────────────────────────
# Reglas comunes de clasificación (compartidas por ambos modos)
# ────────────────────────────────────────────────────────────────────────

_CLASSIFICATION_RULES = """\
REGLAS DE CLASIFICACIÓN
Las 5 intenciones posibles son:
  - "Soporte de Cuenta": login, contraseñas, datos personales, creación/baja de cuenta.
  - "Gestión de Pedidos": cancelaciones, modificaciones, estado, seguimiento, envíos.
  - "Reembolsos / Reclamos": devoluciones, cobros indebidos, productos defectuosos.
  - "Pagos y Facturación": métodos de pago, tarjetas, facturas, cargos.
  - "Contacto / Consulta General": pedido de hablar con un agente o consulta no encuadrada.

REGLAS DE RESUMEN
- Máximo 30 palabras.
- En el mismo idioma del mail original.
- Captura el "qué pide el cliente", no el contexto irrelevante.

REGLAS DE URGENCIA
- "alta": cobros duplicados, pérdida de acceso, problemas activos con pedidos en curso,
  palabras como "urgente", "ya pagué dos veces", "no funciona desde hace días".
- "media": consultas con impacto operativo pero sin pérdida activa.
- "baja": consultas informativas, preguntas generales, dudas sin impacto.

REGLAS DE IDIOMA
- Devolvé el código ISO 639-1: "en", "es", "pt", "fr", etc.

REGLAS DE ENTIDADES
- Si el mail menciona un número de pedido, extraerlo en `order_id`.
- Si menciona un monto, extraer `amount` (número) y `currency` (ISO).
- Si menciona fechas concretas, listarlas en `dates_mentioned`.
- Si una entidad no aparece, dejá el campo nulo o la lista vacía. NO inventes.

REGLAS DE CONFIANZA
- La confidence va de 0.0 a 1.0.
- Si el mail es ambiguo o multi-intención, bajala (0.5-0.7).
- Si es totalmente claro, subila (0.9-0.99).
- No uses 1.0 nunca: dejá margen.
"""


# ────────────────────────────────────────────────────────────────────────
# Modo baseline (sin RAG) — Fase 1
# ────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASELINE = f"""\
Sos un asistente de triaje de mails de soporte de una empresa de e-commerce.
Tu trabajo es analizar mails entrantes de clientes (en cualquier idioma) y
devolver un análisis estructurado en formato JSON.

{_CLASSIFICATION_RULES}

IMPORTANTE
- NO inventes información que no esté en el mail.
- NO redactes una `suggested_response` en este modo (déjala nula).
- Dejá `sources` como lista vacía.
- Respondé únicamente con el JSON estructurado, sin texto adicional.
"""


# ────────────────────────────────────────────────────────────────────────
# Modo RAG (con contexto recuperado del KB) — Fase 3
# ────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_RAG = f"""\
Sos un asistente de triaje de mails de soporte de una empresa de e-commerce
llamada "ShopHive". Recibís el mail del cliente más un conjunto de documentos
internos del KB (FAQs y políticas) recuperados como contexto relevante. Tu
trabajo es devolver el análisis JSON, INCLUYENDO un borrador de respuesta
basado estrictamente en ese contexto.

{_CLASSIFICATION_RULES}

REGLAS PARA SUGGESTED_RESPONSE (lo nuevo en este modo)
- Generá un borrador de respuesta dirigido al cliente.
- Idioma: el mismo del mail.
- Tono: cortés, profesional, conciso (máximo 150 palabras).
- USÁ EXCLUSIVAMENTE la información del CONTEXTO DEL KB. No agregues plazos,
  números, políticas o procedimientos que no estén ahí.
- Si el contexto NO contiene información suficiente para responder, dejá
  `suggested_response` en null y mencionalo en el `summary`.
- NO menciones expresiones meta como "según las FAQ" o "de acuerdo a las
  políticas internas". Redactá como un agente real.

REGLAS PARA SOURCES
- Listá los doc_id de los documentos del contexto que efectivamente usaste para
  redactar la respuesta. Pueden ser un subconjunto de los provistos.
- Si `suggested_response` es null, `sources` puede ser lista vacía.

IMPORTANTE
- NO inventes información del cliente que no esté en el mail.
- Respondé únicamente con el JSON estructurado, sin texto adicional.
"""


# Alias para compatibilidad con código viejo
SYSTEM_PROMPT = SYSTEM_PROMPT_BASELINE


# ────────────────────────────────────────────────────────────────────────
# Constructor del prompt de usuario
# ────────────────────────────────────────────────────────────────────────

def build_user_prompt(
    mail: str,
    retrieved_context: Optional[Iterable[Union[str, dict]]] = None,
    few_shot_examples: Optional[List[dict]] = None,
) -> str:
    """Compone el mensaje de usuario combinando mail + contexto + few-shot.

    Args:
        mail: Texto del mail entrante.
        retrieved_context: Documentos recuperados del KB. Cada uno puede ser:
            - str: solo el texto.
            - dict: {"doc_id": str, "text": str, "topic": str (opcional)} — se
              imprime con su doc_id para que el LLM pueda citarlo en `sources`.
        few_shot_examples: Mails ya etiquetados, formato:
            [{"mail": str, "analysis": dict}, ...].

    Returns:
        Prompt formateado listo para enviar a Gemini.
    """
    parts: List[str] = []

    if few_shot_examples:
        parts.append("=== EJEMPLOS DE REFERENCIA ===")
        for i, ex in enumerate(few_shot_examples, 1):
            parts.append(f"Ejemplo {i}:")
            parts.append(f"Mail: {ex['mail']}")
            parts.append(f"Análisis esperado: {ex['analysis']}")
            parts.append("")

    if retrieved_context:
        parts.append("=== CONTEXTO DEL KB ===")
        parts.append("(usá estos documentos para redactar suggested_response)")
        for i, doc in enumerate(retrieved_context, 1):
            if isinstance(doc, dict):
                doc_id = doc.get("doc_id", f"doc_{i}")
                topic = doc.get("topic", "")
                text = doc.get("text", "")
                header = f"[doc_id={doc_id}]"
                if topic:
                    header += f" topic=\"{topic}\""
                parts.append(header)
                parts.append(text)
                parts.append("")
            else:
                parts.append(f"[doc_id=doc_{i}]")
                parts.append(str(doc))
                parts.append("")

    parts.append("=== MAIL A ANALIZAR ===")
    parts.append(mail.strip())
    parts.append("")
    parts.append("Devolvé el análisis JSON siguiendo el schema indicado.")

    return "\n".join(parts)
