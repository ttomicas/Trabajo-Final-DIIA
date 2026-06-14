# Asistente de Triaje de Mails con RAG y LLM

**Diplomatura en IA Aplicada — Trabajo Final**
Sistema de NLP que clasifica, resume y redacta respuestas a mails de soporte mediante un pipeline híbrido **RAG + LLM (Gemini)**.

## Visión del producto

Para cada mail entrante, en menos de 5 segundos, el sistema devuelve:

1. **Intención clasificada** (5 categorías de negocio).
2. **Resumen breve** del contenido.
3. **Nivel de urgencia** (alta / media / baja).
4. **Entidades extraídas** (n° de pedido, montos, fechas).
5. **Respuesta sugerida** basada en políticas vigentes y casos similares pasados.
6. **Fuentes citadas** del KB que justifican la respuesta (trazabilidad).

## Arquitectura

```
Mail → PII redaction → Embedding (multilingual-e5)
                              ↓
               Chroma retrieval (FAQs + casos pasados)
                              ↓
         Gemini con prompt enriquecido + JSON schema
                              ↓
         Output estructurado → Dashboard del agente
```

## Stack

| Capa | Herramienta |
|---|---|
| LLM | Google Gemini (2.0 Flash + 2.5 Pro) vía AI Studio |
| Embeddings | `intfloat/multilingual-e5-large` |
| Vector store | Chroma (local) |
| PII redaction | Microsoft Presidio + regex |
| Validación de output | Pydantic + Gemini structured output |
| Backend | FastAPI (fase futura) |
| Frontend | Streamlit (fase futura) |
| Mail ingestion | Gmail API / Microsoft Graph (fase futura) |

## Setup

```bash
# 1. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Copiar archivo de entorno
cp .env.example .env
# Editar .env y poner tu GOOGLE_API_KEY (https://aistudio.google.com/app/apikey)
```

## Cómo correrlo

### Notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb       # Exploratory Data Analysis
jupyter notebook notebooks/02_pipeline_demo.ipynb   # Demo del pipeline RAG
```

También se pueden abrir en Google Colab.

## Estructura del repositorio

```
proyecto-final-rag/
├── README.md
├── requirements.txt
├── .env.example                 # plantilla de variables de entorno
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb             # Análisis exploratorio del dataset
│   └── 02_pipeline_demo.ipynb   # Demo end-to-end del RAG (próxima fase)
├── src/
│   ├── __init__.py
│   ├── pii.py                   # redacción de datos sensibles
│   ├── kb_ingest.py             # carga FAQs a Chroma
│   ├── retriever.py             # búsqueda semántica en Chroma
│   ├── prompts.py               # plantillas de prompts
│   ├── llm.py                   # cliente de Gemini
│   ├── schemas.py               # modelos Pydantic del output
│   └── pipeline.py              # orquesta todo el flujo
├── data/
│   ├── kb/                      # FAQs y políticas en Markdown
│   └── examples/                # mails etiquetados para few-shot
└── tests/
    └── test_pipeline.py
```

## Hoja de ruta de desarrollo

| Fase | Entregable | Estado |
|---|---|---|
| 0 | PoC: clasificador NLP con TF-IDF y embeddings | ✅ (repo previo) |
| **1** | **EDA completa del dataset** | **🟢 en este repo** |
| 2 | Pipeline Gemini básico: mail → JSON {intent, summary, urgency, language} | ⏳ próximo |
| 3 | Indexar mini-KB en Chroma + inyectar contexto recuperado | ⏳ |
| 4 | Generar `suggested_response` con Gemini 2.5 Pro | ⏳ |
| 5 | Few-shot examples retrievados de mails ya etiquetados | ⏳ |
| 6 | Dashboard Streamlit para agentes | ⏳ |
| 7 | Integración Gmail / Outlook + feedback loop | ⏳ |

## Dataset

Bitext Customer Support LLM Chatbot Training Dataset (HuggingFace, ~27k muestras, licencia CC BY-NC-SA-4.0). Filtrado a 5 categorías. Detalles y análisis en `notebooks/01_eda.ipynb`.

## Relación con el PoC previo

El PoC entregado anteriormente (clasificador clásico con TF-IDF + LinearSVC) sirve de **baseline de comparación** para este proyecto final. La hipótesis de trabajo es que la combinación clasificador + RAG + LLM agrega valor sobre la clasificación pura en (a) calidad de redacción de respuestas, (b) trazabilidad de decisiones y (c) capacidad multilingüe nativa.
