from pathlib import Path

from logger import get_logger
from .base import DocumentChunk
from .pdf_loader import load_pdf
from .image_loader import load_image, SUPPORTED_EXTENSIONS as IMAGE_EXTS
from .docx_loader import load_docx
from .pptx_loader import load_pptx
from .excel_loader import load_excel
from .markdown_loader import load_markdown
from .web_loader import load_url

log = get_logger("ingestion.router")


def load_document(source: str) -> list[DocumentChunk]:
    log.info("Loading document: %s", source)

    if source.startswith("http://") or source.startswith("https://"):
        log.info("Detected source type: URL")
        chunks = load_url(source)
    else:
        path = Path(source)
        ext = path.suffix.lower()
        log.info("Detected source type: file | extension: %s", ext)

        if ext == ".pdf":
            chunks = load_pdf(source)
        elif ext in IMAGE_EXTS:
            chunks = load_image(source)
        elif ext == ".docx":
            chunks = load_docx(source)
        elif ext == ".pptx":
            chunks = load_pptx(source)
        elif ext in {".xlsx", ".xls", ".csv"}:
            chunks = load_excel(source)
        elif ext in {".md", ".markdown"}:
            chunks = load_markdown(source)
        else:
            log.error("Unsupported file type: %s", ext)
            raise ValueError(f"Unsupported file type: {ext}")

    text_count  = sum(1 for c in chunks if c.chunk_type == "text")
    image_count = sum(1 for c in chunks if c.chunk_type == "image")
    table_count = sum(1 for c in chunks if c.chunk_type == "table")
    log.info(
        "Extraction complete | total=%d  text=%d  image=%d  table=%d",
        len(chunks), text_count, image_count, table_count,
    )
    return chunks
