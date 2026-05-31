import base64
import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from config import IMAGE_MAX_SIZE
from logger import get_logger
from .base import DocumentChunk

log = get_logger("ingestion.pdf")


def load_pdf(path: str) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    source = Path(path).name
    log.info("Opening PDF: %s", source)

    doc = fitz.open(path)
    total_pages = len(doc)
    log.info("PDF has %d page(s)", total_pages)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            chunks.append(DocumentChunk(
                content=text,
                chunk_type="text",
                source=source,
                page=page_num,
                metadata={"file_type": "pdf"},
            ))
            log.debug("Page %d/%d: extracted %d chars of text", page_num, total_pages, len(text))

        images = page.get_images(full=True)
        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img.thumbnail(IMAGE_MAX_SIZE)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()

            chunks.append(DocumentChunk(
                content="",
                chunk_type="image",
                source=source,
                page=page_num,
                metadata={"file_type": "pdf", "image_index": img_index},
                image_b64=b64,
            ))
            log.debug("Page %d/%d: extracted image %d", page_num, total_pages, img_index)

        if images:
            log.info("Page %d/%d: %d image(s) extracted", page_num, total_pages, len(images))

    doc.close()
    log.info("PDF load complete | chunks=%d", len(chunks))
    return chunks
