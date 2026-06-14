"""Cliente de Gemini para análisis estructurado de mails.

Wrapper sobre `google.generativeai` con:
- Carga de credenciales desde `.env` (variable GOOGLE_API_KEY).
- Selección de modelo según tier: "fast" (Gemini 2.0 Flash) o "quality" (2.5 Pro).
- Structured output forzado con response_schema=MailAnalysis (Pydantic).
- Retry con backoff exponencial para errores transitorios (429, 503, network).
- Logging mínimo del uso de tokens y latencia para monitoreo.

Uso típico:
    >>> from src.llm import analyze_mail
    >>> result = analyze_mail("Hi, I need help recovering my password.")
    >>> result.intent
    <Intent.ACCOUNT: 'Soporte de Cuenta'>
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import MailAnalysis

load_dotenv()


_MODEL_TIERS = {
    "fast":    os.getenv("GEMINI_FAST_MODEL",    "gemini-2.0-flash"),
    "quality": os.getenv("GEMINI_QUALITY_MODEL", "gemini-2.5-pro"),
}


class LLMError(RuntimeError):
    """Excepción de alto nivel para errores recuperables o no del cliente LLM."""


def _get_client():
    """Configura y devuelve el módulo genai listo para usar."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "tu_clave_aca":
        raise LLMError(
            "GOOGLE_API_KEY no está seteada. Copiá `.env.example` a `.env` y "
            "completala con tu key de https://aistudio.google.com/app/apikey."
        )
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise LLMError(
            "Falta google-generativeai. Corré: pip install google-generativeai"
        ) from exc
    genai.configure(api_key=api_key)
    return genai


def analyze_mail(
    mail_text: str,
    model_tier: str = "fast",
    retrieved_context: Optional[List[str]] = None,
    few_shot_examples: Optional[List[dict]] = None,
    max_retries: int = 3,
    verbose: bool = False,
) -> MailAnalysis:
    """Analiza un mail con Gemini y devuelve un MailAnalysis validado.

    Args:
        mail_text: Texto del mail. Idealmente con PII ya redactada.
        model_tier: "fast" (default, Gemini 2.0 Flash) o "quality" (Gemini 2.5 Pro).
        retrieved_context: Documentos del KB recuperados vía RAG. None en Fase 1.
        few_shot_examples: Ejemplos para in-context learning. None en Fase 1.
        max_retries: Reintentos ante errores transitorios.
        verbose: Si True, imprime latencia y uso de tokens.

    Returns:
        MailAnalysis con los campos validados por Pydantic.

    Raises:
        LLMError: si la API key falta, el modelo es inválido o se agotan los retries.
    """
    if model_tier not in _MODEL_TIERS:
        raise LLMError(f"model_tier debe ser uno de {list(_MODEL_TIERS)}.")

    genai = _get_client()
    model_name = _MODEL_TIERS[model_tier]
    user_prompt = build_user_prompt(mail_text, retrieved_context, few_shot_examples)

    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": MailAnalysis,
        "temperature": 0.2,
    }

    model = genai.GenerativeModel(
        model_name,
        system_instruction=SYSTEM_PROMPT,
        generation_config=generation_config,
    )

    backoff = 1.0
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        start = time.perf_counter()
        try:
            response = model.generate_content(user_prompt)
            elapsed_ms = (time.perf_counter() - start) * 1000

            parsed: MailAnalysis = response.parsed  # Pydantic ya validado por SDK

            if verbose:
                usage = getattr(response, "usage_metadata", None)
                in_tokens  = getattr(usage, "prompt_token_count",     "?") if usage else "?"
                out_tokens = getattr(usage, "candidates_token_count", "?") if usage else "?"
                print(
                    f"[{model_name}] {elapsed_ms:.0f}ms "
                    f"in={in_tokens}tok out={out_tokens}tok"
                )

            return parsed

        except Exception as exc:  # noqa: BLE001 — SDK lanza varios tipos
            last_error = exc
            if verbose:
                print(f"  Intento {attempt + 1}/{max_retries} falló: {exc!s}")
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2

    raise LLMError(
        f"Falló después de {max_retries} reintentos. Último error: {last_error!s}"
    ) from last_error
