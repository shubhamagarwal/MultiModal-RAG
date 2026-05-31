# MultiModal RAG — Architecture & Technical Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Tech Stack](#2-tech-stack)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Project Structure](#4-project-structure)
5. [End-to-End Flow](#5-end-to-end-flow)
   - 5.1 [Ingestion Flow](#51-ingestion-flow)
   - 5.2 [Query Flow](#52-query-flow)
6. [Module Deep-Dives](#6-module-deep-dives)
   - 6.1 [Ingestion Layer](#61-ingestion-layer)
   - 6.2 [Processing Layer](#62-processing-layer)
   - 6.3 [Embedding Layer](#63-embedding-layer)
   - 6.4 [Vector Store](#64-vector-store)
   - 6.5 [Retrieval Layer](#65-retrieval-layer)
   - 6.6 [Generation Layer](#66-generation-layer)
   - 6.7 [Evaluation Layer (RAGAS)](#67-evaluation-layer-ragas)
   - 6.8 [Security Layer (PII Guard)](#68-security-layer-pii-guard)
   - 6.9 [Logging](#69-logging)
   - 6.10 [UI Layer (Chainlit)](#610-ui-layer-chainlit)
7. [Data Model](#7-data-model)
8. [Configuration Reference](#8-configuration-reference)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Known Limitations](#10-known-limitations)

---

## 1. Overview

**MultiModal RAG** is a Retrieval-Augmented Generation system that lets users chat with their documents. It can ingest text, tables, and images from multiple document formats, store them in a local vector database, and answer natural-language questions using GPT-4o — always citing the exact source document. A built-in PII security layer automatically detects and redacts personal information during ingestion and blocks queries containing PII.

**Core capabilities:**
- Ingest PDF, DOCX, PPTX, Excel/CSV, images (PNG/JPG/TIFF), and web URLs
- Describe images using GPT-4o Vision and embed their descriptions alongside text
- **PII Security Guardrails** — detect and redact personal information (names, emails, SSNs, etc.) using Microsoft Presidio; toggleable via the UI settings panel
- Retrieve the most semantically similar chunks using cosine similarity
- Generate answers that cite specific chunks using `[1]`, `[2]` notation
- Show only the sources actually referenced in the answer
- Evaluate pipeline quality with RAGAS metrics
- **Structured logging** across all modules for observability and debugging

---

## 2. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Chainlit 2.x | Browser chat interface with file upload + settings panel |
| **LLM (generation)** | GPT-4o via GitHub Models | Answer synthesis + strict citation |
| **LLM (vision)** | GPT-4o via OpenAI API | Image-to-text description |
| **Embeddings** | `text-embedding-3-large` | Semantic vector representation |
| **Vector Store** | ChromaDB (persistent, local) | HNSW cosine similarity search |
| **PII Detection** | Microsoft Presidio (Analyzer + Anonymizer) | Detect and redact personal information |
| **NLP (PII)** | spaCy `en_core_web_lg` | Named Entity Recognition backend for Presidio |
| **PDF parsing** | PyMuPDF (`fitz`) | Text extraction + embedded image extraction |
| **DOCX parsing** | python-docx | Paragraphs, tables, inline images |
| **PPTX parsing** | python-pptx | Slide text and picture shapes |
| **Excel/CSV** | pandas + openpyxl + tabulate | Sheets rendered as markdown tables |
| **Web scraping** | requests + BeautifulSoup4 | HTML → clean text |
| **Text splitting** | LangChain `RecursiveCharacterTextSplitter` | Overlap-aware chunking |
| **Evaluation** | RAGAS 0.1.21 | Faithfulness, answer relevancy, precision, recall |
| **Logging** | Python `logging` (custom `logger.py`) | Structured timestamped logs across all modules |
| **Config** | python-dotenv | Environment variable management |

---

## 3. Architecture Diagram

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CHAINLIT UI                              │
│         (browser chat + file upload + settings panel)           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Settings: PII Security Guardrails  [ON / OFF]      │        │
│  └─────────────────────────────────────────────────────┘        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │ File Upload / URL / Query  │
              └──────────┬────────────────┘
                         │
          ┌──────────────▼──────────────────────────────┐
          │              app.py (Orchestrator)           │
          │  Routes between: ingest / query / evaluate   │
          │  Manages: PII toggle, conversation history   │
          └──┬──────────────────────────────────────┬───┘
             │                                      │
    ┌────────▼─────────┐                  ┌─────────▼────────┐
    │  INGESTION FLOW  │                  │   QUERY FLOW     │
    └────────┬─────────┘                  └─────────┬────────┘
             │                                      │
    ┌────────▼──────────────────┐        ┌──────────▼───────────┐
    │     router.py             │        │  PII GUARD (query)   │
    │  (dispatch by ext/URL)    │        │  check_query(text)   │
    └─┬──┬──┬──┬──┬──┬─────────┘        │  BLOCK if PII found  │
      │  │  │  │  │  │                  └──────────┬───────────┘
     PDF IMG DOCX PPTX XLS URL                     │
      │  │  │  │  │  │                  ┌──────────▼───────────┐
    ┌─▼──▼──▼──▼──▼──▼──────────┐       │    retriever.py       │
    │   DocumentChunk objects    │       │  score threshold 0.35 │
    │  {content, type, source,   │       └──────────┬───────────┘
    │   page, metadata, b64}     │                  │
    └────────────┬───────────────┘       ┌──────────▼───────────┐
                 │                       │   chroma_store.py     │
    ┌────────────▼───────────────┐       │  cosine ANN search   │
    │      pipeline.py           │       └──────────┬───────────┘
    │  text → split_text_chunk   │                  │
    │  image → describe_image    │       ┌──────────▼───────────┐
    │  table → pass-through      │       │  openai_embedder.py  │
    └────────────┬───────────────┘       │  embed query text    │
                 │                       └──────────┬───────────┘
    ┌────────────▼───────────────┐                  │
    │  PII GUARD (ingestion)     │       ┌──────────▼───────────┐
    │  pii_guard.redact()        │       │   generator.py        │
    │  <PERSON>, <EMAIL>, etc.   │       │  build_context [1..N] │
    │  (when guardrails ON)      │       │  GPT-4o strict cite   │
    └────────────┬───────────────┘       │  returns (answer,     │
                 │                       │   cited_chunks)       │
    ┌────────────▼───────────────┐       └──────────┬───────────┘
    │   openai_embedder.py       │                  │
    │  text-embedding-3-large    │       ┌──────────▼───────────┐
    │  batched (100 texts/call)  │       │  RAGAS EVALUATION    │
    └────────────┬───────────────┘       │  (async background)  │
                 │                       │                      │
    ┌────────────▼───────────────┐       │  faithfulness        │
    │   chroma_store.py          │       │  answer_relevancy    │
    │  upsert with MD5 chunk ID  │       │  context_precision*  │
    │  persistent cosine HNSW    │       │  context_recall*     │
    └────────────────────────────┘       │  (* needs ground     │
                                         │     truth)           │
                                         └──────────┬───────────┘
                                                    │
                                         ┌──────────▼───────────┐
                                         │  Score table shown   │
                                         │  in Chainlit UI      │
                                         │  below each answer   │
                                         └──────────────────────┘

    ┌────────────────────────────────────────────────────────────┐
    │                  CROSS-CUTTING CONCERNS                    │
    ├────────────────────────────────────────────────────────────┤
    │  logger.py     │  Structured logging (HH:MM:SS | LEVEL)  │
    │  config.py     │  Environment vars + derived constants    │
    │  pii_guard.py  │  Presidio-based PII detect / redact     │
    └────────────────────────────────────────────────────────────┘
```

### Component Interaction (Sequence)

```
User                  Chainlit            app.py          Modules
 │                       │                  │                │
 │── open chat ──────────▶                  │                │
 │                       │── on_start ─────▶│                │
 │                       │                  │── init settings (PII toggle ON)
 │                       │                  │── count chunks ▶ ChromaStore
 │                       │◀── welcome msg ──│                │
 │                       │                  │                │
 │── toggle PII OFF ─────▶                  │                │
 │                       │── on_settings ──▶│                │
 │                       │                  │── update pii_guard_enabled=False
 │                       │◀── "PII OFF" ───│                │
 │                       │                  │                │
 │── upload file ────────▶                  │                │
 │                       │── on_message ───▶│                │
 │                       │                  │── load_document▶ router
 │                       │                  │                │── pdf/docx/..
 │                       │                  │◀─ DocumentChunk[]
 │                       │                  │── process_chunks▶ pipeline
 │                       │                  │                │── chunker
 │                       │                  │                │── image_processor (GPT-4o Vision)
 │                       │                  │                │── pii_guard.redact() (if PII ON)
 │                       │                  │◀─ processed chunks (PII-redacted)
 │                       │                  │── add_chunks ──▶ ChromaStore
 │                       │                  │                │── embed_texts (text-embedding-3-large)
 │                       │                  │                │── chromadb.upsert
 │                       │◀── "✓ ingested" ─│                │
 │                       │                  │                │
 │── ask question ───────▶                  │                │
 │                       │── on_message ───▶│                │
 │                       │                  │── check_query()▶ pii_guard (if PII ON)
 │                       │                  │                │── detect PII entities
 │                       │                  │                │── BLOCK if found
 │                       │                  │◀─ pass / PIIBlockedError
 │                       │                  │── retrieve ────▶ retriever
 │                       │                  │                │── embed query
 │                       │                  │                │── chroma.query (n*2)
 │                       │                  │                │── filter score ≥ 0.35
 │                       │                  │◀─ filtered chunks (sorted desc)
 │                       │                  │── generate_answer▶ generator
 │                       │                  │                │── build_context [1..N]
 │                       │                  │                │── GPT-4o (strict citations)
 │                       │                  │                │── extract cited [N] indices
 │                       │                  │◀─ (answer, cited_chunks)
 │                       │◀── answer + sources
 │◀── response ──────────│                  │                │
 │                       │                  │                │
 │                       │                  │  ┌─────────────────────────────────┐
 │                       │                  │  │  ThreadPoolExecutor (async)     │
 │                       │                  │──┼─▶ _run_ragas()                  │
 │                       │                  │  │    ragas_eval.evaluate_rag()    │
 │                       │                  │  │    ├─ faithfulness              │
 │                       │                  │  │    ├─ answer_relevancy          │
 │                       │                  │  │    ├─ context_precision*        │
 │                       │                  │  │    └─ context_recall*           │
 │                       │                  │  └──────────────┬──────────────────┘
 │                       │                  │◀─ ragas scores  │
 │                       │◀── score table ──│                  │
 │◀── RAGAS metrics ─────│                  │                  │
 │  (updates in-place    │                  │                  │
 │   below the answer)   │                  │                  │
```

> `*` context_precision and context_recall require `ground_truth` in the sample.

---

## 4. Project Structure

```
MultiModal-RAG/
│
├── app.py                          # Chainlit app — orchestrates all flows
├── config.py                       # All environment variables + derived constants
├── logger.py                       # Centralised structured logging (timestamp | level | module)
├── requirements.txt                # Pinned Python dependencies
├── chainlit.md                     # Welcome message shown in the chat UI
├── .env.example                    # Template for environment variables
│
├── src/
│   ├── ingestion/
│   │   ├── base.py                 # DocumentChunk dataclass
│   │   ├── router.py               # Dispatches to correct loader by extension/URL
│   │   ├── pdf_loader.py           # PyMuPDF: text per page + base64 images
│   │   ├── image_loader.py         # Pillow: PNG/JPG/TIFF → base64
│   │   ├── docx_loader.py          # python-docx: paragraphs + tables + images
│   │   ├── pptx_loader.py          # python-pptx: slide text + picture shapes
│   │   ├── excel_loader.py         # pandas: sheets → markdown tables
│   │   └── web_loader.py           # requests + BS4: HTML → clean text
│   │
│   ├── processing/
│   │   ├── pipeline.py             # Orchestrates chunking + image description + PII redaction
│   │   ├── chunker.py              # RecursiveCharacterTextSplitter (1000 tok, 200 overlap)
│   │   └── image_processor.py      # GPT-4o Vision → text description (with refusal guard)
│   │
│   ├── embeddings/
│   │   └── openai_embedder.py      # text-embedding-3-large, batched 100/call
│   │
│   ├── vectorstore/
│   │   └── chroma_store.py         # ChromaDB: upsert, cosine query, list, delete
│   │
│   ├── retrieval/
│   │   └── retriever.py            # Query + score threshold filter (≥ 0.35)
│   │
│   ├── generation/
│   │   └── generator.py            # GPT-4o with [N] citation prompt + citation parser
│   │
│   ├── security/
│   │   └── pii_guard.py            # Presidio PII detection, redaction, and query blocking
│   │
│   └── evaluation/
│       └── ragas_eval.py           # RAGAS: faithfulness, relevancy, precision, recall
│
└── data/
    ├── uploads/                    # Temp storage for uploaded files
    └── chroma_db/                  # Persistent ChromaDB on disk (HNSW index)
```

---

## 5. End-to-End Flow

### 5.1 Ingestion Flow

```
INPUT: file path or URL
         │
         ▼
┌─────────────────────┐
│  router.py          │  Inspects extension or URL scheme
│  load_document()    │  Returns list[DocumentChunk]
└──────────┬──────────┘
           │
    ┌──────┴───────────────────────────────────────────┐
    │ chunk_type = "text"    chunk_type = "image"       │
    │                                                   │
    ▼                                                   ▼
chunker.py                                 image_processor.py
RecursiveCharacterTextSplitter              GPT-4o Vision API
  chunk_size=1000, overlap=200              → detailed text description
  splits on: \n\n → \n → ". " → " "        stored in chunk.content
    │                                               │
    └──────────────────────┬────────────────────────┘
                           │
                           ▼
                  ┌─────────────────────┐
                  │  PII GUARD          │  (when guardrails ON)
                  │  pii_guard.redact() │
                  │  Presidio Analyzer  │
                  │  + Anonymizer       │
                  │                     │
                  │  "John Smith" →     │
                  │  "<PERSON>"         │
                  │                     │
                  │  "john@x.com" →     │
                  │  "<EMAIL_ADDRESS>"  │
                  └─────────┬───────────┘
                            │
                            ▼
                  openai_embedder.py
                  text-embedding-3-large
                  batch size = 100 texts
                           │
                           ▼
                  chroma_store.add_chunks()
                  - ID = MD5(source|page|content[:100])
                  - upsert (safe re-ingest)
                  - stores: embedding, content, metadata
                  - persists to ./data/chroma_db
```

**What each loader extracts:**

| Loader | Text | Tables | Images |
|---|---|---|---|
| `pdf_loader` | Per-page text via PyMuPDF | — | Base64 via `extract_image()` |
| `image_loader` | — | — | Base64 via Pillow |
| `docx_loader` | All paragraphs joined | Rows as `col1 \| col2` | Inline rels |
| `pptx_loader` | Per-slide text from all shapes | — | `shape_type == 13` |
| `excel_loader` | — | Markdown table per sheet via `df.to_markdown()` | — |
| `web_loader` | Boilerplate-stripped body text | — | — |

### 5.2 Query Flow

```
INPUT: natural-language question
         │
         ▼
┌──────────────────────────┐
│  PII GUARD (query-time)  │  (when guardrails ON)
│  check_query(text)       │
│  detect() → PIIFinding[] │
│                          │
│  PII found?              │
│    YES → PIIBlockedError │  → "Query blocked" message to user
│    NO  → continue ↓      │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  retriever.retrieve()    │
│  1. embed query text     │  → text-embedding-3-large (1 API call)
│  2. chroma.query(n*2)    │  → fetch 12 candidates
│  3. filter score ≥ 0.35  │  → drop irrelevant chunks
│  4. sort desc, take top 6│
└─────────────┬────────────┘
              │  list[dict]: {content, metadata, score}
              ▼
┌──────────────────────────┐
│  generator.build_context │
│  Label each chunk [1]..[N]│
│  Include source + score   │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  GPT-4o (temp=0.1)       │
│  System: cite every claim │
│  as [N], don't hallucinate│
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  _extract_cited_indices() │
│  regex: \[(\d+,...)\]     │
│  maps [2,3] → chunks[1,2] │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  app._format_sources()   │
│  Only cited_chunks shown  │
│  Deduped by source|page   │
└─────────────┬────────────┘
              │
              ▼
     answer + cited sources
     displayed in Chainlit
              │
              │  (non-blocking — runs in ThreadPoolExecutor)
              ▼
┌──────────────────────────────────────────┐
│  RAGAS EVALUATION  (ragas_eval.py)       │
│                                          │
│  Input:                                  │
│    question  → user query                │
│    answer    → GPT-4o response           │
│    contexts  → cited chunk contents      │
│                                          │
│  Metrics computed by LLM judge (GPT-4o): │
│    faithfulness      — claims grounded?  │
│    answer_relevancy  — on-topic answer?  │
│    context_precision — chunks relevant?* │
│    context_recall    — all chunks found?*│
│                                          │
│  Output: score table (0.0 – 1.0)        │
│  updates in-place below the answer       │
└──────────────────────────────────────────┘

OUTPUT: answer + sources (immediate)
        RAGAS score table (async, ~5-10s later)

* requires ground_truth in sample
```

---

## 6. Module Deep-Dives

### 6.1 Ingestion Layer

**`src/ingestion/base.py` — DocumentChunk**

The universal data carrier throughout the pipeline:

```python
@dataclass
class DocumentChunk:
    content: str          # text (or GPT-4o image description after processing)
    chunk_type: Literal["text", "image", "table"]
    source: str           # filename or URL — used for source citation
    page: int             # page number (slide number for PPTX)
    metadata: dict        # file_type, sheet, slide, image_index, etc.
    image_b64: str | None # raw base64 JPEG — passed to GPT-4o Vision
```

**`src/ingestion/router.py`** — single entry point `load_document(source)`. Dispatches by:
- URL prefix → `load_url()`
- `.pdf` → `load_pdf()`
- `.png/.jpg/.jpeg/.tiff/.tif/.webp/.bmp/.gif` → `load_image()`
- `.docx` → `load_docx()`
- `.pptx` → `load_pptx()`
- `.xlsx/.xls/.csv` → `load_excel()`

### 6.2 Processing Layer

**`src/processing/pipeline.py`** — routes each chunk by type:
- `image` → `describe_image()` (GPT-4o Vision) → fills `chunk.content`
- `text` / `table` → `split_text_chunk()` (splits if > 1000 chars)
- **PII redaction** (when `redact_pii=True`): calls `pii_guard.redact()` on each chunk after splitting/describing, replacing detected PII entities with `<ENTITY_TYPE>` placeholders
- Filters out any chunk with empty content after processing

**`src/processing/chunker.py`** — `RecursiveCharacterTextSplitter`:
- Chunk size: **1000 characters**
- Overlap: **200 characters** (preserves cross-boundary context)
- Separator cascade: `\n\n` → `\n` → `. ` → ` ` → `""` (prefers semantic breaks)

**`src/processing/image_processor.py`** — Vision pipeline with three guards:
1. `VISION_ENABLED` check — skips API call if no `OPENAI_API_KEY` (GitHub Models doesn't support base64 image data URIs)
2. Refusal detection — checks response for phrases like "cannot process", "I'm sorry"
3. Exception handler — catches API errors, stores a human-readable placeholder

### 6.3 Embedding Layer

**`src/embeddings/openai_embedder.py`**

- Model: `text-embedding-3-large` (3072 dimensions)
- Batch size: **100 texts per API call** (avoids rate limits)
- 100ms sleep between batches
- Empty strings replaced with `"empty"` to avoid API errors
- Returns `list[list[float]]` aligned 1:1 with input texts

### 6.4 Vector Store

**`src/vectorstore/chroma_store.py`**

| Operation | Implementation |
|---|---|
| Storage | `chromadb.PersistentClient` → `./data/chroma_db` |
| Distance | Cosine (HNSW index via `metadata={"hnsw:space": "cosine"}`) |
| Chunk ID | `MD5(source + page + content[:100])` — deterministic, prevents duplicates |
| Write | `collection.upsert()` — safe to re-ingest same document |
| Read | `collection.query(embeddings, n_results)` → converts distance to score: `score = 1 - distance` |
| Score range | 0.0 (unrelated) → 1.0 (identical) |

### 6.5 Retrieval Layer

**`src/retrieval/retriever.py`**

```
retrieve(query, n_results=6):
  1. chroma.query(n_results * 2)   ← fetch 12 to have headroom
  2. filter score >= 0.35          ← MIN_RELEVANCE_SCORE
  3. sort descending by score
  4. return top n_results          ← max 6 chunks
```

The `MIN_RELEVANCE_SCORE = 0.35` threshold is configurable via `.env`. Lower it (e.g., 0.2) for broader recall; raise it (e.g., 0.5) for stricter precision.

### 6.6 Generation Layer

**`src/generation/generator.py`**

**System prompt design** (key constraints):
```
- Each chunk is labelled [1], [2], [3]...
- Cite every claim using its chunk number
- If answer not present → respond with exact phrase "The answer is not available..."
- Do NOT use knowledge outside the context
```

**Citation extraction:**
```python
def _extract_cited_indices(answer: str) -> set[int]:
    # Matches [1], [2,3], [1, 2, 3]
    re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', answer)
```

**Return value:** `(answer: str, cited_chunks: list[dict])`
- `cited_chunks` contains only the retrieved chunks whose index appears in the answer
- If the model says the answer isn't available, `cited_chunks` is empty → no sources shown

### 6.7 Evaluation Layer (RAGAS)

**`src/evaluation/ragas_eval.py`**

RAGAS evaluates the RAG pipeline quality using an LLM judge. Input format:

```python
samples = [
    {
        "question":    "What is the error rate?",
        "answer":      "The error rate is 5% [1].",
        "contexts":    ["...chunk from xls...", "...chunk from pdf..."],
        "ground_truth": "The error rate is 5%."   # optional
    }
]
results = evaluate_rag(samples)
```

| Metric | Requires Ground Truth | What it measures |
|---|---|---|
| `faithfulness` | No | Is every claim in the answer supported by a retrieved chunk? |
| `answer_relevancy` | No | Does the answer actually address the question asked? |
| `context_precision` | Yes | Of retrieved chunks, what fraction are genuinely relevant? |
| `context_recall` | Yes | Were all chunks needed to answer the question retrieved? |

Scores are floats 0–1. Higher is better for all metrics.

### 6.8 Security Layer (PII Guard)

**`src/security/pii_guard.py`**

PII (Personally Identifiable Information) protection powered by **Microsoft Presidio**, operating in two modes:

| Mode | Function | When Used | Behaviour |
|---|---|---|---|
| **Redaction** | `redact(text)` | During ingestion (pipeline.py) | Replaces PII with `<ENTITY_TYPE>` placeholders |
| **Blocking** | `check_query(text)` | At query time (app.py) | Raises `PIIBlockedError` if PII detected |

**Watched entity types (12):**

| Entity | Example |
|---|---|
| `PERSON` | John Smith |
| `EMAIL_ADDRESS` | john@example.com |
| `PHONE_NUMBER` | +1-555-123-4567 |
| `CREDIT_CARD` | 4111-1111-1111-1111 |
| `US_SSN` | 123-45-6789 |
| `US_ITIN` | 900-70-1234 |
| `US_PASSPORT` | A12345678 |
| `US_DRIVER_LICENSE` | D123-456-789 |
| `IBAN_CODE` | DE89370400440532013000 |
| `IP_ADDRESS` | 192.168.1.1 |
| `MEDICAL_LICENSE` | DEA-AB1234567 |
| `NRP` | Nationality/Religious/Political group |

**Configuration:**
- Confidence threshold: **0.70** (entities below this score are ignored)
- NLP backend: spaCy `en_core_web_lg` (loaded lazily on first use)
- Engines are singleton-initialized (`_analyzer`, `_anonymizer`)

**How redaction works:**
```
Input:  "John Smith's SSN is 123-45-6789 and email is john@x.com"
Output: "<PERSON>'s SSN is <US_SSN> and email is <EMAIL_ADDRESS>"
```

**How query blocking works:**
```
User query:  "What is John Smith's salary?"
→ detect() finds PERSON entity with score > 0.70
→ PIIBlockedError raised
→ app.py catches and displays: "Query blocked — personal information detected."
→ Query is NEVER forwarded to the LLM
```

**Toggle:** Users can enable/disable PII guardrails via the Chainlit **Settings** panel. When OFF, documents are stored as-is and all queries pass through unchecked.

### 6.9 Logging

**`logger.py`**

Centralised structured logging used by every module in the system:

```python
get_logger(name: str) -> logging.Logger
```

- **Format:** `HH:MM:SS | LEVEL    | module.name                         | message`
- **Level:** INFO (stdout only, filtered to exclude higher levels)
- **Propagation:** disabled (`propagate = False`) to avoid duplicate output
- Loggers are cached — calling `get_logger("x")` twice returns the same instance

**Usage pattern:** every module calls `log = get_logger("module.name")` at import time and uses structured log messages with key=value pairs for machine-parseable output.

**Log output examples:**
```
14:23:01 | INFO     | processing.pipeline              | ┌─ [PROCESSING] Pipeline started | raw_chunks=12 | PII redaction=ON ─┐
14:23:01 | INFO     | security.pii_guard               |   [PII GUARD] Redacted 2 item(s):
14:23:01 | INFO     | security.pii_guard               |     PERSON                 | confidence=0.85 | value='J***n'
14:23:02 | INFO     | vectorstore.chroma               | Upsert complete | stored=12 | total_in_db=48
```

### 6.10 UI Layer (Chainlit)

**`app.py`** handles three distinct message types:

| Input | Detection | Action |
|---|---|---|
| File attachment | `message.elements` not empty | Ingest → process (+ PII redact) → embed → store |
| URL | `text.startswith("http")` | Scrape → process (+ PII redact) → embed → store |
| Question | Everything else | PII check → Retrieve → generate → display answer + cited sources |

**Settings panel:**
- **PII Security Guardrails** toggle (ON by default)
- `on_settings_update` handler persists the toggle in `cl.user_session`
- When toggled, a confirmation message is shown to the user

**PII guard integration:**
- **Ingestion:** `process_chunks(raw, redact_pii=pii_on)` — the `redact_pii` flag flows from the UI toggle
- **Query:** `check_query(text)` is called before retrieval; catches `PIIBlockedError` and shows a user-friendly block message with tips to rephrase

Conversation history (last 10 turns = 20 messages) is stored in `cl.user_session` and passed to the generator to support follow-up questions.

---

## 7. Data Model

```
DocumentChunk (in-memory, during ingestion)
├── content: str
├── chunk_type: "text" | "image" | "table"
├── source: str
├── page: int
├── metadata: dict
└── image_b64: str | None

ChromaDB Document (persisted)
├── id: str (MD5 hash)
├── embedding: list[float]   ← 3072-dim from text-embedding-3-large
├── document: str            ← chunk.content
└── metadata:
    ├── source: str
    ├── page: int
    ├── chunk_type: str
    ├── file_type: str
    ├── has_image: "true"    ← only if image_b64 was present
    └── ...loader-specific fields (sheet, slide, image_index, url, title)

Retrieved Result (query output)
├── content: str
├── score: float             ← 1 - cosine_distance (0..1)
└── metadata: dict           ← same as ChromaDB metadata
```

---

## 8. Configuration Reference

All settings live in `config.py`, loaded from `.env`:

| Variable | Default | Description |
|---|---|---|
| `GITHUB_TOKEN` | — | GitHub Models token (text generation + embeddings) |
| `OPENAI_API_KEY` | — | Direct OpenAI key (required for image/vision processing) |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | Model for generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |
| `OPENAI_VISION_MODEL` | `gpt-4o` | Model for image description |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Where ChromaDB stores its index |
| `UPLOAD_DIR` | `./data/uploads` | Temp directory for uploaded files |
| `CHUNK_SIZE` | `1000` | Max characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `MAX_RETRIEVAL_RESULTS` | `6` | Max chunks returned per query |
| `MIN_RELEVANCE_SCORE` | `0.35` | Minimum cosine similarity to include a chunk |
| `IMAGE_MAX_SIZE` | `(1024, 1024)` | Max pixel dimensions for image resizing |

**API routing logic:**

```
GITHUB_TOKEN set?
  YES → API_KEY = GITHUB_TOKEN, BASE_URL = https://models.inference.ai.azure.com
  NO  → API_KEY = OPENAI_API_KEY, BASE_URL = None (direct OpenAI)

VISION_ENABLED = bool(OPENAI_API_KEY)
  → Vision always uses direct OpenAI (GitHub Models doesn't support base64 image URIs)
```

**PII Guard settings (not in `.env` — configured in code):**

| Setting | Value | Location |
|---|---|---|
| Confidence threshold | `0.70` | `pii_guard.py` |
| Watched entity types | 12 types | `pii_guard.py` |
| Default toggle state | `ON` | `app.py` (Chainlit settings panel) |
| spaCy model | `en_core_web_lg` | Auto-loaded by Presidio |

---

## 9. Key Design Decisions

**Why base64 for images rather than file URLs?**
Images extracted from PDFs/DOCX have no persistent URL. Base64 lets us pass them directly to the Vision API without a hosting service. The raw base64 is kept in `image_b64` but only the text description is embedded and stored.

**Why MD5 for chunk IDs?**
Deterministic hashing of `(source, page, content[:100])` makes upsert idempotent — re-ingesting the same document doesn't create duplicates.

**Why fetch `n_results * 2` then filter?**
ChromaDB's `n_results` is a hard cap. If we ask for 6 and 4 are below threshold, we'd return only 2. Fetching 12 and filtering gives us up to 6 valid results while discarding noise.

**Why `[N]` citation format in the prompt?**
Asking the model to use numbered citations ties the answer back to specific retrieved chunks — making source attribution deterministic (regex-parseable) rather than heuristic (string-matching source names, which can hallucinate).

**Why temperature=0.1 for generation?**
The retrieval+citation flow requires consistent, predictable output format. Low temperature avoids the model paraphrasing its citations in ways that break the regex.

**Why RAGAS 0.1.21 specifically?**
Later versions (0.4.x) introduced a dependency on `langchain_community.chat_models.vertexai` which requires Google Cloud credentials. 0.1.21 is the last stable version without this transitive dependency.

**Why Microsoft Presidio for PII detection?**
Presidio is open-source, runs locally (no data leaves the machine), supports pluggable NLP backends (spaCy), and covers a broad set of entity types out of the box. It avoids sending raw PII to external APIs for detection.

**Why two PII modes (redact vs. block)?**
Ingested documents may contain PII that needs to be stored in anonymised form (redaction), while user queries containing PII should be rejected entirely (blocking) to prevent leaking personal data to the LLM. Separating the modes lets each pathway apply the appropriate guardrail.

**Why is PII guard toggleable?**
Not all use cases require PII protection (e.g., internal demos, non-sensitive documents). A runtime toggle via the settings panel lets users opt out without restarting the server. The default is ON (secure by default).

**Why a custom logger module?**
Standard Python logging with a consistent format across all modules ensures structured, machine-parseable output. The INFO-only filter on stdout keeps the console clean while preserving the ability to add file handlers or higher-level handlers later.

---

## 10. Known Limitations

| Limitation | Workaround |
|---|---|
| GitHub Models doesn't support base64 image URIs | Set `OPENAI_API_KEY` alongside `GITHUB_TOKEN` to enable vision |
| No `.doc` (old Word), `.odt`, `.txt`, `.html` (local) support | Convert to supported format before uploading |
| Images in Excel files not extracted | Excel images are rare in practice; not yet implemented |
| Score threshold (0.35) may be too strict for short documents | Lower `MIN_RELEVANCE_SCORE` in `.env` |
| ChromaDB is local — no multi-user persistence | Switch to Qdrant or Pinecone for production use |
| Streaming stops mid-response if the network drops | No automatic retry; the user must re-ask the question |
| PII detection is English-only (`en_core_web_lg`) | Add spaCy models for other languages and configure multi-language Presidio |
| PII confidence threshold (0.70) is not configurable via `.env` | Edit `CONFIDENCE_THRESHOLD` in `pii_guard.py` |
| First query with PII guard ON has cold-start latency | spaCy model loads lazily on first `detect()`/`redact()` call (~2-3s) |
| PII redaction modifies chunk content before hashing | Re-ingesting with PII guard toggled creates separate chunk IDs |
