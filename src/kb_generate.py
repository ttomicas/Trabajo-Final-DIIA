"""Generador sintético del knowledge base (KB) usando Gemini.

Para el PoC necesitamos un corpus de FAQs y políticas de empresa que el
sistema RAG pueda consultar. En lugar de inventarlas a mano, las hacemos
generar por Gemini con prompts controlados y luego las revisamos.

Output: archivos Markdown en `data/kb/` con frontmatter YAML para
trazabilidad (categoría, tema, idioma).

Uso:
    >>> from src.kb_generate import generate_all_faqs
    >>> generate_all_faqs(output_dir="data/kb", verbose=True)
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from .llm import _MODEL_TIERS, _get_client


# 30 temas, 6 por categoría × 5 categorías. Cubre la mayoría de las preguntas
# reales de un mail de soporte de e-commerce. La lista se diseña para que
# cada categoría tenga "señal léxica" diversa.
TOPICS: List[Tuple[str, str, str]] = [
    # (category, topic_slug, topic_human_readable)

    # ACCOUNT
    ("ACCOUNT", "password-reset",          "How to reset a forgotten password"),
    ("ACCOUNT", "account-locked",          "What to do if my account got locked"),
    ("ACCOUNT", "email-verification",      "Email verification process"),
    ("ACCOUNT", "update-personal-info",    "How to update personal information"),
    ("ACCOUNT", "delete-account",          "How to delete or close my account"),
    ("ACCOUNT", "two-factor-auth",         "Setting up two-factor authentication"),

    # ORDER
    ("ORDER", "track-order",               "How to track the status of an order"),
    ("ORDER", "cancel-order",              "How to cancel an order before shipment"),
    ("ORDER", "modify-order",              "Modifying an order before it ships"),
    ("ORDER", "order-not-arrived",         "What to do if my order didn't arrive"),
    ("ORDER", "wrong-item-received",       "I received the wrong item — what now"),
    ("ORDER", "shipping-options",          "Available shipping options and times"),

    # REFUND
    ("REFUND", "refund-policy",            "Refund policy and timeframes"),
    ("REFUND", "how-to-request-refund",    "Step by step to request a refund"),
    ("REFUND", "defective-product-return", "Returning a defective product"),
    ("REFUND", "duplicate-charge",         "What to do if I was charged twice"),
    ("REFUND", "partial-refund",           "When partial refunds apply"),
    ("REFUND", "refund-credited-where",    "Where the refund money is credited"),

    # PAYMENT
    ("PAYMENT", "accepted-payment-methods","Accepted payment methods"),
    ("PAYMENT", "update-credit-card",      "How to update my credit card on file"),
    ("PAYMENT", "payment-declined",        "What to do if my payment was declined"),
    ("PAYMENT", "download-invoice",        "How to download my invoice"),
    ("PAYMENT", "currency-conversion",     "Currency conversion and FX charges"),
    ("PAYMENT", "subscription-billing",    "How subscription billing works"),

    # CONTACT
    ("CONTACT", "customer-service-hours",  "Customer service operating hours"),
    ("CONTACT", "speak-to-human",          "How to speak with a human agent"),
    ("CONTACT", "email-contact",           "Email channels for support"),
    ("CONTACT", "phone-support",           "Phone support availability"),
    ("CONTACT", "live-chat",               "Live chat availability and how to use"),
    ("CONTACT", "response-time-sla",       "Typical response time SLA"),
]


PROMPT_TEMPLATE = """\
Sos un redactor de FAQs para una empresa ficticia de e-commerce llamada "ShopHive".
Tu trabajo es escribir UNA FAQ corta y útil sobre el siguiente tema.

Tema: {topic}
Categoría: {category}

REGLAS
- Idioma: inglés.
- Longitud: 80-180 palabras.
- Estructura: 1-2 párrafos cortos. Si corresponde, una lista de 2-4 puntos.
- Tono: claro, profesional, amable. NO uses lenguaje de marketing ni emojis.
- Contenido: incluí pasos concretos cuando aplique. Mencioná tiempos, plazos,
  canales de contacto y políticas relevantes.
- Inventá datos realistas: por ejemplo, "el reintegro tarda entre 5 y 10 días
  hábiles", "horario de atención de 9 a 18", etc. Sé internamente consistente.
- NO incluyas título — solo el cuerpo de la FAQ.
- NO inventes URLs ni emails reales (usá placeholders genéricos si hace falta:
  "support@shophive.example" o "shophive.example/account").

Escribí ahora la FAQ.
"""


def _slugify(text: str) -> str:
    """Convierte una cadena en un slug seguro para nombre de archivo."""
    s = text.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s


def _build_markdown(category: str, topic_slug: str, topic: str,
                    body: str, model: str) -> str:
    """Arma el archivo Markdown final con frontmatter de trazabilidad."""
    return (
        f"---\n"
        f"category: {category}\n"
        f"topic_slug: {topic_slug}\n"
        f"topic: {topic}\n"
        f"language: en\n"
        f"generated_by: {model}\n"
        f"---\n\n"
        f"# {topic}\n\n"
        f"{body.strip()}\n"
    )


def generate_one_faq(category: str, topic: str, model_tier: str = "fast",
                     verbose: bool = False) -> str:
    """Genera el cuerpo de una sola FAQ. Devuelve el texto plano."""
    from google.genai import types  # type: ignore

    client = _get_client()
    model_name = _MODEL_TIERS[model_tier]
    prompt = PROMPT_TEMPLATE.format(topic=topic, category=category)

    config = types.GenerateContentConfig(
        temperature=0.6,  # algo creativa pero no descontrolada
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )
    text = response.text.strip()
    if verbose:
        usage = getattr(response, "usage_metadata", None)
        out_tokens = getattr(usage, "candidates_token_count", "?") if usage else "?"
        print(f"  [{model_name}] {out_tokens} tokens generados")
    return text


def generate_all_faqs(
    output_dir: str = "data/kb",
    sleep_between_calls_s: float = 5.0,
    skip_existing: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """Genera las 30 FAQs y las guarda en `output_dir` como .md.

    Args:
        output_dir: Carpeta donde se guardan los .md.
        sleep_between_calls_s: Pausa entre llamadas para respetar rate limits.
        skip_existing: Si True, no regenera archivos que ya existen.
        verbose: Imprime progreso.

    Returns:
        Dict {ruta_archivo: cuerpo_de_la_faq}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_name = _MODEL_TIERS["fast"]
    generated: Dict[str, str] = {}

    for i, (category, topic_slug, topic) in enumerate(TOPICS, 1):
        filename = f"{category.lower()}_{topic_slug}.md"
        filepath = out / filename

        if skip_existing and filepath.exists():
            if verbose:
                print(f"[{i:02d}/{len(TOPICS)}] {filename}  (ya existe, salteado)")
            generated[str(filepath)] = filepath.read_text(encoding="utf-8")
            continue

        if verbose:
            print(f"[{i:02d}/{len(TOPICS)}] {category} — {topic}")
        try:
            body = generate_one_faq(category, topic, verbose=verbose)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️  Falló: {exc!s}")
            time.sleep(sleep_between_calls_s)
            continue

        md = _build_markdown(category, topic_slug, topic, body, model_name)
        filepath.write_text(md, encoding="utf-8")
        generated[str(filepath)] = md

        if i < len(TOPICS):
            time.sleep(sleep_between_calls_s)

    if verbose:
        print(f"\n✅ {len(generated)} FAQs generadas en {out}")
    return generated
