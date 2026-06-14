"""Redacción de datos sensibles (PII) antes de enviar al LLM.

Combina dos enfoques:
- Regex para patrones comunes (mail, teléfono, números de tarjeta, n° doc).
- Microsoft Presidio para detección estadística más amplia (nombres,
  ubicaciones, IBANs, etc.).

Las entidades detectadas se reemplazan por tokens genéricos para que el
LLM no procese datos personales.

TODO (Fase 2): implementar redact_text() usando presidio_analyzer +
presidio_anonymizer.
"""
from __future__ import annotations


def redact_text(text: str) -> str:
    """Reemplaza PII en `text` por tokens genéricos.

    Por implementar en la Fase 2 del proyecto.
    """
    raise NotImplementedError("Pendiente — Fase 2.")
