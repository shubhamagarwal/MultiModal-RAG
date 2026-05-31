from pathlib import Path

from logger import get_logger
from .base import DocumentChunk

log = get_logger("ingestion.markdown_loader")


def load_markdown(source: str) -> list[DocumentChunk]:
    path = Path(source)
    text = path.read_text(encoding="utf-8", errors="replace")
    log.info("Markdown loaded | file=%s | chars=%d", path.name, len(text))

    if not text.strip():
        log.info("Markdown file is empty, returning no chunks")
        return []

    return [DocumentChunk(
        content=text,
        chunk_type="text",
        source=path.name,
        page=1,
        metadata={"source": path.name, "page": 1, "chunk_type": "text"},
    )]
