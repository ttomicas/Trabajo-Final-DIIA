"""Modelos Pydantic para validar la salida del LLM.

Estos schemas se le pasan a Gemini vía `response_schema` en structured output,
garantizando que el modelo siempre devuelva JSON válido y parseable.

Notas de compatibilidad con Gemini:
- Solo se usan tipos primitivos (str, float, int, bool), Enum, List y Optional.
- Sin dict libre ni Union complejos: Gemini no los maneja bien.
- Los nombres de campo van en snake_case.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """Las 5 categorías de negocio del proyecto."""
    ACCOUNT = "Soporte de Cuenta"
    ORDER = "Gestión de Pedidos"
    REFUND = "Reembolsos / Reclamos"
    PAYMENT = "Pagos y Facturación"
    CONTACT = "Contacto / Consulta General"


class Urgency(str, Enum):
    HIGH = "alta"
    MEDIUM = "media"
    LOW = "baja"


class Entities(BaseModel):
    """Entidades extraídas del mail. Todas opcionales; el LLM las completa si aparecen."""
    order_id: Optional[str] = Field(
        None, description="Número de pedido si se menciona en el mail."
    )
    amount: Optional[float] = Field(
        None, description="Monto numérico si se menciona en el mail."
    )
    currency: Optional[str] = Field(
        None, description="Moneda detectada en código ISO (USD, EUR, ARS...)."
    )
    dates_mentioned: List[str] = Field(
        default_factory=list, description="Fechas mencionadas en el mail."
    )


class MailAnalysis(BaseModel):
    """Salida estructurada del pipeline para un mail entrante.

    Esta es la "contract" del sistema: cualquier cliente que consuma el output
    puede depender de que estos campos siempre estén presentes y tipados.
    """
    intent: Intent = Field(..., description="Categoría de intención clasificada.")
    confidence: float = Field(
        ...,
        description="Confianza del modelo en la clasificación entre 0.0 y 1.0."
    )
    summary: str = Field(
        ..., description="Resumen del mail en 1-2 líneas, máximo 30 palabras."
    )
    urgency: Urgency = Field(..., description="Nivel de urgencia inferido.")
    language: str = Field(
        ..., description="Código ISO 639-1 del idioma detectado (en, es, pt...)."
    )
    entities: Entities = Field(
        default_factory=Entities, description="Entidades extraídas del mail."
    )
    suggested_response: Optional[str] = Field(
        None, description="Borrador de respuesta. Se completa solo en Fase 3+ (RAG)."
    )
    sources: List[str] = Field(
        default_factory=list,
        description="IDs de los documentos del KB usados como contexto (Fase 3+)."
    )
