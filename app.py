import asyncio
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Switch

from config import UPLOAD_DIR
from logger import get_logger
from src.ingestion import load_document
from src.processing import process_chunks
from src.retrieval.retriever import retrieve, get_store
from src.generation import generate_answer
from src.generation.generator import stream_answer
from src.security.pii_guard import check_query, PIIBlockedError, redact

log = get_logger("app")
_executor = ThreadPoolExecutor(max_workers=2)

_PII_GUARD_KEY = "pii_guard_enabled"
_TENANT_KEY = "tenant_id"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pii_enabled() -> bool:
    return cl.user_session.get(_PII_GUARD_KEY, True)


def _tenant_id() -> str:
    return cl.user_session.get(_TENANT_KEY, "default")


# ── Auth ──────────────────────────────────────────────────────────────────────

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    # Identity-only login — username becomes the tenant ID.
    # No password validation: suitable for internal/demo use.
    # Replace with real credential checks for production.
    if username.strip():
        log.info("[AUTH] Login | user=%s", username)
        return cl.User(identifier=username, metadata={"role": "user"})
    return None


def _format_sources(cited_chunks: list[dict]) -> str:
    if not cited_chunks:
        return "_No sources cited — answer may not be in the knowledge base._"

    seen, lines = set(), []
    for item in cited_chunks:
        meta = item["metadata"]
        key = f"{meta.get('source')}|{meta.get('page')}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- **{meta.get('source', 'unknown')}** "
            f"(page {meta.get('page', '?')}, "
            f"type: {meta.get('chunk_type', 'text')}, "
            f"relevance: {item.get('score', 0):.0%})"
        )
    return "\n".join(lines)


def _score_bar(score: float) -> str:
    filled = round(score * 10)
    return f"`{'█' * filled}{'░' * (10 - filled)}` {score:.2f}"


def _run_ragas(question: str, answer: str, contexts: list[str]) -> dict:
    log.info("RAGAS thread started | contexts=%d", len(contexts))
    try:
        from src.evaluation.ragas_eval import evaluate_rag
        scores = dict(evaluate_rag([{
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }]))
        log.info("RAGAS thread complete | scores=%s",
                 {k: f"{v:.3f}" for k, v in scores.items() if isinstance(v, float)})
        return scores
    except Exception as e:
        log.error("RAGAS thread failed: %s", e)
        return {"error": str(e)}


def _format_ragas(results: dict) -> str:
    if "error" in results:
        return f"RAGAS evaluation failed: `{results['error']}`"

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    lines = ["### RAGAS Evaluation", "",
             "| Metric | Score | Bar |", "|---|---|---|"]
    for key, label in metric_labels.items():
        if key in results:
            score = float(results[key])
            lines.append(f"| {label} | `{score:.3f}` | {_score_bar(score)} |")

    lines += ["", "_Faithfulness: answer grounded in context · "
              "Answer Relevancy: answer addresses the question_"]
    return "\n".join(lines)


# ── Chainlit lifecycle ────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    user = cl.user_session.get("user")
    tenant = user.identifier if user else "default"
    cl.user_session.set(_TENANT_KEY, tenant)
    log.info("Chat session started | tenant=%s", tenant)

    # Register the settings panel with a PII toggle (ON by default)
    settings = await cl.ChatSettings([
        Switch(
            id=_PII_GUARD_KEY,
            label="PII Security Guardrails",
            initial=True,
            description=(
                "When ON: redacts PII from ingested documents and blocks queries "
                "that contain personal information (names, emails, SSNs, etc.)."
            ),
        )
    ]).send()

    cl.user_session.set(_PII_GUARD_KEY, settings[_PII_GUARD_KEY])
    cl.user_session.set("history", [])

    store = get_store(tenant)
    sources = store.list_sources()
    log.info("Knowledge base | tenant=%s | chunks=%d  sources=%d", tenant, store.count(), len(sources))

    await cl.Message(
        content=(
            f"**MultiModal RAG** ready — workspace: `{tenant}`\n\n"
            f"Knowledge base: **{store.count()} chunks** from **{len(sources)} source(s)**.\n\n"
            "**Upload files** (PDF, DOCX, PPTX, XLSX, CSV, PNG, JPG, MD) or **paste a URL** to ingest, "
            "then ask any question.\n\n"
            "> Use the **Settings** panel to toggle PII Security Guardrails on or off."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    enabled = settings[_PII_GUARD_KEY]
    cl.user_session.set(_PII_GUARD_KEY, enabled)

    if enabled:
        log.info("┌─ [PII GUARD] Guardrails switched ON by user ─────────────────────────┐")
        log.info("│  Ingestion : PII will be redacted before storing chunks              │")
        log.info("│  Query     : requests containing PII will be blocked                 │")
        log.info("└───────────────────────────────────────────────────────────────────────┘")
        await cl.Message(
            content=(
                "**PII Security Guardrails: ON**\n\n"
                "- Ingested documents will have PII automatically redacted.\n"
                "- Queries containing personal information will be blocked."
            )
        ).send()
    else:
        log.info("┌─ [PII GUARD] Guardrails switched OFF by user ────────────────────────┐")
        log.info("│  Ingestion : documents stored as-is (no PII scanning)                │")
        log.info("│  Query     : all queries forwarded to LLM without PII check          │")
        log.info("└───────────────────────────────────────────────────────────────────────┘")
        await cl.Message(
            content=(
                "**PII Security Guardrails: OFF**\n\n"
                "PII detection and redaction is disabled. "
                "Documents and queries will be processed as-is."
            )
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    history: list[dict] = cl.user_session.get("history", [])
    tenant = _tenant_id()
    store = get_store(tenant)
    pii_on = _pii_enabled()

    # ── 1. File uploads ───────────────────────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if not isinstance(element, cl.File):
                continue

            dest = Path(UPLOAD_DIR) / element.name
            shutil.copy(element.path, dest)
            log.info("File upload received: %s | pii_guard=%s", element.name, "ON" if pii_on else "OFF")
            await cl.Message(content=f"Ingesting **{element.name}**...").send()

            try:
                log.info("[INGESTION] Loading document: %s", element.name)
                raw = load_document(str(dest))
                log.info("[INGESTION] Raw chunks extracted: %d", len(raw))

                await cl.Message(
                    content=f"Processing {len(raw)} chunk(s) from **{element.name}**..."
                ).send()

                log.info("[PROCESSING] Starting pipeline | pii_redaction=%s",
                         "ON" if pii_on else "OFF")
                processed = process_chunks(raw, redact_pii=pii_on)
                log.info("[PROCESSING] Processed chunks: %d", len(processed))

                log.info("[VECTORSTORE] Storing chunks for: %s", element.name)
                added = store.add_chunks(processed)
                log.info("[VECTORSTORE] Stored %d chunk(s) | total_in_db=%d", added, store.count())

                shield = " (PII redacted)" if pii_on else " (PII guardrails off)"
                await cl.Message(
                    content=f"✓ **{element.name}** ingested — {added} chunk(s) added{shield}."
                ).send()
            except Exception as e:
                log.error("Ingestion failed for %s: %s", element.name, e, exc_info=True)
                await cl.Message(content=f"Error ingesting **{element.name}**: {e}").send()

        if not message.content.strip():
            return

    # ── 2. URL ingestion ──────────────────────────────────────────────────────
    text = message.content.strip()
    if text.startswith("http://") or text.startswith("https://"):
        log.info("[INGESTION] URL received: %s", text)
        await cl.Message(content=f"Fetching **{text}**...").send()
        try:
            raw = load_document(text)
            log.info("[INGESTION] URL extracted %d chunk(s)", len(raw))
            processed = process_chunks(raw, redact_pii=pii_on)
            log.info("[PROCESSING] URL processed -> %d chunk(s)", len(processed))
            added = store.add_chunks(processed)
            log.info("[VECTORSTORE] URL stored %d chunk(s)", added)
            await cl.Message(content=f"✓ URL ingested — {added} chunk(s) added.").send()
        except Exception as e:
            log.error("URL ingestion failed: %s | error: %s", text, e, exc_info=True)
            await cl.Message(content=f"Error ingesting URL: {e}").send()
        return

    # ── 3. RAG query ──────────────────────────────────────────────────────────
    if not text:
        return

    if store.count() == 0:
        await cl.Message(
            content="The knowledge base is empty. Please upload a document or paste a URL first."
        ).send()
        return

    # ── PII guardrail on query (only when enabled) ────────────────────────────
    if pii_on:
        log.info("┌─ [PII GUARD] Scanning query for PII ─────────────────────────────────┐")
        log.info("│  query='%s...'", text[:60])
        try:
            check_query(text)
            log.info("└─ [PII GUARD] Scan passed — query forwarded to retrieval ─────────────┘")
        except PIIBlockedError as e:
            log.info("└─ [PII GUARD] Query BLOCKED — not forwarded to LLM ──────────────────┘")
            await cl.Message(
                content=(
                    "**Query blocked — personal information detected.**\n\n"
                    f"{e}\n\n"
                    "_Tip: rephrase without names, emails, phone numbers, or SSNs. "
                    "You can also turn off PII Guardrails in Settings._"
                )
            ).send()
            return
    else:
        log.info("[PII GUARD] Skipped — guardrails are OFF")

    log.info("[RETRIEVAL] Query: '%s...' | tenant=%s", text[:60], tenant)
    await cl.Message(content="Searching knowledge base...").send()
    retrieved = retrieve(text, tenant_id=tenant)
    log.info("[RETRIEVAL] Retrieved %d chunk(s) after threshold filtering", len(retrieved))

    if not retrieved:
        await cl.Message(
            content="No sufficiently relevant context found. Try rephrasing or uploading more documents."
        ).send()
        return

    # Redact PII from retrieved chunks when guard is ON — covers docs ingested with guard OFF
    if pii_on:
        log.info("┌─ [PII GUARD] Post-retrieval redaction pass ───────────────────────────────┐")
        redacted_retrieved = []
        for chunk in retrieved:
            clean_content, findings = redact(chunk["content"])
            if findings:
                types = sorted({f.entity_type for f in findings})
                log.info("│  [PII GUARD] source=%-30s | redacted types=%s",
                         chunk["metadata"].get("source", "?"), types)
            redacted_chunk = dict(chunk)
            redacted_chunk["content"] = clean_content
            redacted_retrieved.append(redacted_chunk)
        retrieved = redacted_retrieved
        log.info("└─ [PII GUARD] Post-retrieval redaction complete ────────────────────────────┘")

    log.info("[GENERATION] Streaming answer from %d chunk(s)", len(retrieved))
    answer_msg = cl.Message(content="")
    await answer_msg.send()

    gen = stream_answer(text, retrieved, history=history)
    answer = ""
    cited_chunks: list[dict] = []
    try:
        while True:
            token = next(gen)
            answer += token
            await answer_msg.stream_token(token)
    except StopIteration as exc:
        cited_chunks = exc.value or []

    log.info("[GENERATION] Stream complete | cited_sources=%d | answer_len=%d",
             len(cited_chunks), len(answer))

    sources_md = _format_sources(cited_chunks)
    answer_msg.content = f"{answer}\n\n---\n**Sources:**\n{sources_md}"
    await answer_msg.update()

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history[-20:])

    # ── 4. RAGAS evaluation (async, non-blocking) ─────────────────────────────
    log.info("[RAGAS] Scheduling evaluation in background thread")
    ragas_msg = await cl.Message(content="_Running RAGAS evaluation..._").send()

    eval_chunks = cited_chunks if cited_chunks else retrieved
    contexts = [c["content"] for c in eval_chunks]
    log.info("[RAGAS] Using %d context chunk(s) for evaluation", len(contexts))

    loop = asyncio.get_event_loop()
    ragas_results = await loop.run_in_executor(
        _executor, _run_ragas, text, answer, contexts
    )

    log.info("[RAGAS] Evaluation complete — updating UI")
    ragas_msg.content = _format_ragas(ragas_results)
    await ragas_msg.update()
