from logger import get_logger
from src.ingestion.base import DocumentChunk
from src.security.pii_guard import redact
from .chunker import split_text_chunk
from .image_processor import describe_image

log = get_logger("processing.pipeline")


def _redact_chunk(chunk: DocumentChunk) -> DocumentChunk:
    clean, findings = redact(chunk.content)
    if findings:
        types = sorted({f.entity_type for f in findings})
        log.info("  [PII GUARD] source=%-30s page=%-3s | redacted types=%s",
                 chunk.source, chunk.page, types)
        chunk.content = clean
    return chunk


def process_chunks(
    raw_chunks: list[DocumentChunk],
    progress_cb=None,
    redact_pii: bool = True,
) -> list[DocumentChunk]:
    total = len(raw_chunks)
    log.info("┌─ [PROCESSING] Pipeline started | raw_chunks=%d | PII redaction=%s ─┐",
             total, "ON" if redact_pii else "OFF")

    text_chunks  = [c for c in raw_chunks if c.chunk_type == "text"]
    image_chunks = [c for c in raw_chunks if c.chunk_type == "image"]
    table_chunks = [c for c in raw_chunks if c.chunk_type == "table"]
    log.info("Input breakdown | text=%d  image=%d  table=%d",
             len(text_chunks), len(image_chunks), len(table_chunks))

    processed: list[DocumentChunk] = []

    for i, chunk in enumerate(raw_chunks):
        if progress_cb:
            progress_cb(i, total, chunk)

        if chunk.chunk_type == "image":
            log.info("[%d/%d] Describing image | source=%s page=%s",
                     i + 1, total, chunk.source, chunk.page)
            chunk = describe_image(chunk)
            if redact_pii:
                chunk = _redact_chunk(chunk)
            processed.append(chunk)

        else:
            sub_chunks = split_text_chunk(chunk)
            if redact_pii:
                sub_chunks = [_redact_chunk(c) for c in sub_chunks]
            processed.extend(sub_chunks)

    before_filter = len(processed)
    processed = [c for c in processed if c.content.strip()]
    dropped = before_filter - len(processed)
    if dropped:
        log.info("Dropped %d empty chunk(s) after processing", dropped)

    log.info("└─ [PROCESSING] Pipeline complete | output_chunks=%d ──────────────────┘", len(processed))
    return processed
