# MultiModal RAG

A production-ready Retrieval-Augmented Generation system that ingests and queries across multiple document types using OpenAI + ChromaDB.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         CHAINLIT UI                            │
│              browser chat · file upload · settings             │
│         ┌────────────────────────────────────────────┐         │
│         │  Auth login  →  username = tenant ID        │         │
│         │  Settings: PII Security Guardrails [ON/OFF] │         │
│         └────────────────────────────────────────────┘         │
└───────────────────────────┬────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │   File / URL / Question     │
              └──────┬─────────────┬────────┘
                     │             │
          ┌──────────▼──┐    ┌─────▼──────────────────────────┐
          │  INGESTION  │    │           QUERY FLOW            │
          │             │    │                                 │
          │ PDF         │    │  1. PII Guard — block if PII    │
          │ PNG/JPG/TIFF│    │     detected in query text      │
          │ DOCX        │    │                                 │
          │ PPTX        │    │  2. Retriever — embed query     │
          │ XLSX / CSV  │    │     → ChromaDB cosine search    │
          │ Markdown    │    │     → filter score ≥ 0.35       │
          │ Web URL     │    │                                 │
          └──────┬──────┘    │  3. Generator — GPT-4o with    │
                 │           │     [N] citations, temp=0.1     │
          ┌──────▼──────┐    │     streaming token-by-token    │
          │  Processing │    │                                 │
          │             │    │  4. RAGAS Eval (async)          │
          │ text chunks │    │     faithfulness · relevancy    │
          │ GPT-4o      │    │     precision · recall          │
          │  Vision     │    │                                 │
          │ PII redact  │    └─────────────────────────────────┘
          └──────┬──────┘
                 │
          ┌──────▼──────────────────────────────────┐
          │  OpenAI Embeddings  (text-embedding-3-large, 3072-dim) │
          └──────┬──────────────────────────────────┘
                 │
          ┌──────▼──────────────────────────────────┐
          │  ChromaDB  (per-tenant isolated collections)          │
          │                                                       │
          │  rag_alice ──── Alice's private knowledge base        │
          │  rag_bob   ──── Bob's private knowledge base          │
          │  rag_...   ──── (one collection per user)             │
          └───────────────────────────────────────────────────────┘
```

> Full component diagrams, sequence flows, module deep-dives and design rationale are in [ARCHITECTURE.md](./ARCHITECTURE.md).

## Supported Document Types

| Format | What's extracted |
|--------|-----------------|
| PDF | Text per page + embedded images (GPT-4o described) |
| PNG/JPG/TIFF | Full image description via GPT-4o Vision |
| DOCX | Paragraphs, tables, inline images |
| PPTX | Slide text + images |
| XLSX/CSV | Tabular data as markdown tables |
| MD / Markdown | Full document text |
| Web URL | Scraped page text |

## Features

### Multi-Tenancy
Each user gets a fully isolated knowledge base. When you open the app, a login screen prompts for a username — that username becomes your tenant ID. Documents you ingest are stored in a dedicated ChromaDB collection (`rag_{username}`) and are never visible to other users.

```
ChromaDB
├── collection: rag_alice       ← Alice's private knowledge base
│   ├── doc_a.pdf chunks
│   └── report.xlsx chunks
└── collection: rag_bob         ← Bob's private knowledge base
    └── presentation.pptx chunks
```

> The default login accepts any username with any password (identity-only). To enforce real credentials, replace the `auth_callback` in `app.py` with your own validation logic.

### PII Security Guardrails
Microsoft Presidio scans documents and queries for personal information. Toggle on/off via the **Settings** panel in the UI.

- **Ingestion**: PII is redacted from document chunks before storing (names, emails, SSNs, phone numbers, credit cards, etc.)
- **Query**: Queries containing PII are blocked before reaching the LLM
- **Post-retrieval**: Even if documents were ingested with the guard off, enabling it before querying will redact PII from retrieved chunks before they reach the generator

### Streaming Responses
Answers stream token-by-token into the UI as the LLM generates them — no waiting for the full response before seeing output.

### RAGAS Evaluation
After every query, four RAG quality metrics are computed asynchronously and displayed inline:

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are the retrieved chunks relevant to the question? |
| Context Recall | Did retrieval capture all necessary information? |

### Citation-Based Sources
Only sources the LLM actually cited in its answer are shown — not every retrieved chunk.

## Setup

```bash
# 1. Clone and install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download spaCy model (required for PII detection)
python -m spacy download en_core_web_lg

# 3. Configure
cp .env.example .env
# Edit .env — add OPENAI_API_KEY or GITHUB_TOKEN

# 4. Generate Chainlit auth secret (required for multi-tenancy login)
chainlit create-secret
# Copy the printed CHAINLIT_AUTH_SECRET=... line into your .env

# 5. Run
chainlit run app.py
```

Then open http://localhost:8000 in your browser. You will be prompted to log in — enter any username to create your private workspace.

## API Keys

The app supports two key configurations:

| Mode | Required variables | Vision support |
|------|--------------------|----------------|
| OpenAI direct | `OPENAI_API_KEY` | Yes (GPT-4o) |
| GitHub Models | `GITHUB_TOKEN` | No (text only) |

Set `OPENAI_API_KEY` for full multimodal support including image description. Use `GITHUB_TOKEN` for a free-tier GitHub Models endpoint (images will produce a placeholder description).

## Project Structure

```
├── app.py                  # Chainlit UI + orchestration
├── config.py               # All configuration
├── src/
│   ├── ingestion/          # Document loaders per file type
│   │   ├── router.py       # Auto-routes by extension/URL
│   │   ├── pdf_loader.py
│   │   ├── image_loader.py
│   │   ├── docx_loader.py
│   │   ├── pptx_loader.py
│   │   ├── excel_loader.py
│   │   ├── markdown_loader.py
│   │   └── web_loader.py
│   ├── processing/
│   │   ├── pipeline.py     # Orchestrates chunking + image description + PII redaction
│   │   ├── chunker.py      # Recursive text splitter (chunk=1000, overlap=200)
│   │   └── image_processor.py  # GPT-4o Vision → text description
│   ├── embeddings/
│   │   └── openai_embedder.py  # text-embedding-3-large, batched
│   ├── vectorstore/
│   │   └── chroma_store.py     # Per-tenant ChromaDB collections, cosine HNSW
│   ├── retrieval/
│   │   └── retriever.py        # Tenant-scoped query, score threshold filtering
│   ├── generation/
│   │   └── generator.py        # GPT-4o streaming answer synthesis + citation extraction
│   ├── evaluation/
│   │   └── ragas_eval.py       # RAGAS 4-metric async evaluation
│   └── security/
│       └── pii_guard.py        # Presidio PII detection, redaction, query blocking
└── data/
    ├── uploads/            # Temp storage for uploaded files
    └── chroma_db/          # Persisted vector store (all tenants)
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (enables vision) |
| `GITHUB_TOKEN` | — | GitHub Models token (used if no OpenAI key) |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | Model for generation + image description |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `UPLOAD_DIR` | `./data/uploads` | File upload directory |
| `MIN_RELEVANCE_SCORE` | `0.35` | Minimum cosine similarity to include a chunk |
| `MAX_RETRIEVAL_RESULTS` | `6` | Maximum chunks returned per query |

## Production Roadmap

See [PRODUCTION_ROADMAP.md](./PRODUCTION_ROADMAP.md) for a full list of planned improvements across security, performance, reliability, and infrastructure.
