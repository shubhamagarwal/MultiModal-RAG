import base64
import io
from pathlib import Path

from PIL import Image

from config import IMAGE_MAX_SIZE
from logger import get_logger
from .base import DocumentChunk

log = get_logger("ingestion.image")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp", ".gif"}


def load_image(path: str) -> list[DocumentChunk]:
    ext = Path(path).suffix.lower()
    source = Path(path).name
    log.info("Loading image: %s (format: %s)", source, ext)

    if ext not in SUPPORTED_EXTENSIONS:
        log.error("Unsupported image format: %s", ext)
        raise ValueError(f"Unsupported image format: {ext}")

    img = Image.open(path).convert("RGB")
    original_size = img.size
    img.thumbnail(IMAGE_MAX_SIZE)
    log.debug("Image resized: %s -> %s", original_size, img.size)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    log.info("Image load complete | size=%dx%d | base64_len=%d", img.size[0], img.size[1], len(b64))

    return [
        DocumentChunk(
            content="",
            chunk_type="image",
            source=source,
            page=1,
            metadata={"file_type": "image"},
            image_b64=b64,
        )
    ]
