# Production Readiness Roadmap

This document outlines features and modules to evolve the MultiModal RAG system from a working prototype into a production-grade application, organised by concern.

---

## Security & Compliance

| Feature | Why | Tool / Approach |
|---|---|---|
| Authentication & RBAC | Control who can query/ingest | Chainlit built-in auth + JWT |
| Rate limiting | Prevent abuse and cost overrun | Token bucket per user session |
| Audit logging | Track who queried what and when | Structured JSON logs → SIEM |
| Secrets management | No API keys in env files | AWS Secrets Manager / HashiCorp Vault |
| Input sanitization | Block prompt injection attacks | Guardrails AI / NeMo Guardrails |
| Output filtering | Block harmful LLM responses | OpenAI content moderation API |

> PII guardrails (Microsoft Presidio) are already implemented with a UI toggle. The above items extend the security surface beyond PII.

---

## Performance & Scalability

| Feature | Why | Tool / Approach |
|---|---|---|
| Semantic caching | Skip redundant LLM calls for similar queries | GPTCache / Redis + cosine similarity |
| Async ingestion queue | Non-blocking uploads for large documents | Celery + Redis / RQ |
| Reranking | Better retrieval precision after vector search | Cohere Rerank / BGE-Reranker |
| Hybrid search | BM25 + vector for keyword + semantic matching | Migrate to Qdrant or Weaviate |
| Streaming responses | Faster perceived latency | OpenAI `stream=True` + Chainlit streaming |
| Embedding cache | Skip re-embedding unchanged chunks | MD5 content hash before embed (partial foundation already in vectorstore) |

---

## Reliability & Observability

| Feature | Why | Tool / Approach |
|---|---|---|
| Structured tracing | End-to-end request tracing per query | OpenTelemetry + Jaeger |
| LLM observability | Token usage, latency, cost per query | Langfuse / LangSmith / Helicone |
| Health checks | Readiness and liveness for deployment | `/health` endpoint returning KB stats |
| Persistent RAGAS metrics | Track answer quality over time | Write scores to PostgreSQL / InfluxDB |
| Alerting | Detect PII block spikes, error rate surges | Prometheus + Grafana |
| Retry logic | Handle transient OpenAI API failures | `tenacity` with exponential backoff |

> RAGAS evaluation is already wired in per-query (async, non-blocking). Persisting scores to a database enables quality trending.

---

## Data & Knowledge Management

| Feature | Why | Tool / Approach |
|---|---|---|
| Document versioning | Track re-ingested documents, avoid stale chunks | Metadata version field + soft-delete |
| Incremental ingestion | Only re-embed changed pages | Content hash comparison before upsert |
| Multi-tenancy | Separate knowledge bases per user or org | ChromaDB collection per tenant |
| Chunk deduplication | Prevent duplicate chunks from repeated uploads | MD5 hash on content before store |
| Knowledge base expiry | Auto-purge outdated documents | TTL metadata field + scheduled cleanup job |

---

## UX & Conversational Quality

| Feature | Why | Tool / Approach |
|---|---|---|
| Query rewriting | Improve retrieval for vague or short queries | HyDE (Hypothetical Document Embeddings) |
| Long-session memory | Context-aware multi-turn beyond 20 messages | Summarisation of older history turns |
| Feedback collection | Thumbs up / down per answer for ground truth | Chainlit feedback API → RAGAS dataset |
| Fallback to web search | Answer when knowledge base has no match | Tavily / Serper API |
| Citation rendering | Clickable source links in the UI | Chainlit `Text` elements with file URLs |

> Conversation history (last 20 messages) is already maintained per session. Feedback collection would close the loop for continuous quality improvement.

---

## Infrastructure & Deployment

| Feature | Why | Tool / Approach |
|---|---|---|
| Containerisation | Reproducible, portable deploys | Docker + docker-compose |
| Production vector DB | ChromaDB is single-node — scale out | Qdrant Cloud / Pinecone / Weaviate |
| CI/CD pipeline | Automated tests and linting on every PR | GitHub Actions |
| Load testing | Know the breaking point before production | Locust |
| Horizontal scaling | Run multiple app instances | Kubernetes + stateless session design |

---

## Recommended Immediate Wins

These deliver the highest return for the least effort and should be prioritised first.

| Priority | Feature | Effort | Impact |
|---|---|---|---|
| 1 | **Reranking** (Cohere / BGE) | Low — 2-file change | Biggest retrieval quality boost |
| 2 | **Streaming responses** | Low — 1-file change | Feels significantly faster to users |
| 3 | **Langfuse observability** | Low — drop-in SDK | Free, full LLM cost + latency visibility |
| 4 | **Retry logic** (`tenacity`) | Low — wrapper around OpenAI calls | Prevents random failures under load |
| 5 | **Docker** | Medium — Dockerfile + compose | Unlocks any deployment target |
| 6 | **Semantic caching** | Medium | Direct cost reduction on repeated queries |

---

## Current Production-Ready Features

Already implemented in this codebase:

- Multimodal ingestion (PDF, DOCX, PPTX, XLSX, CSV, images, web URLs)
- PII detection and redaction (Presidio) with UI toggle — at ingestion and query time
- Query-time PII redaction for chunks ingested before guard was enabled
- Relevance score filtering (`MIN_RELEVANCE_SCORE=0.35`)
- Citation-based source attribution (only cited sources shown)
- Async RAGAS evaluation (faithfulness, answer relevancy, context precision, context recall)
- Structured INFO-only logging across all modules
- Conversation history (last 20 turns)
- MD5-based chunk deduplication in the vector store
