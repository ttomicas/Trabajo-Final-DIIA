"""Plantillas de prompts para Gemini.

Las separamos en su propio módulo para iterar sobre ellas sin tocar el resto
del pipeline. El SYSTEM_PROMPT es estable; build_prompt() compone el mensaje
de usuario combinando mail + contexto recuperado (RAG) + few-shot examples.
"""
from __future__ import annotations

from typing import List, Optional


SYSTEM_PROMPT = """\
Sos un asistente de triaje de mails de soporte de una empresa de e-commerce.
Tu trabajo es analizar mails entrantes de clientes (en cualquier idioma) y
devolver un análisis estructurado en formato JSON.

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

IMPORTANTE
- NO inventes información que no esté en el mail.
- NO redactes una `suggested_response` en este modo (déjala nula).
- Respondé únicamente con el JSON estructurado, sin texto adicional.
"""


def build_user_prompt(
    mail: str,
    retrieved_context: Optional[List[str]] = None,
    few_shot_examples: Optional[List[dict]] = None,
) -> str:
    """Compone el mensaje de usuario completo para Gemini.

    Args:
        mail: Texto del mail entrante (ya redactado, sin PII).
        retrieved_context: Documentos recuperados del KB vía RAG. None en Fase 1.
        few_shot_examples: Ejemplos de mails ya etiquetados, formato:
            [{"mail": str, "analysis": dict}, ...]. None en Fase 1.

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
        parts.append("=== CONTEXTO RELEVANTE DEL KB ===")
        for i, doc in enumerate(retrieved_context, 1):
            parts.append(f"[Doc {i}]: {doc}")
        parts.append("")

    parts.append("=== MAIL A ANALIZAR ===")
    parts.append(mail.strip())
    parts.append("")
    parts.append("Devolvé el análisis JSON siguiendo el schema indicado.")

    return "\n".join(parts)
