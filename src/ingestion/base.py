from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DocumentChunk:
    content: str                          # text content (or image description)
    chunk_type: Literal["text", "image", "table"]
    source: str                           # original file path or URL
    page: int = 0
    metadata: dict = field(default_factory=dict)
    image_b64: str | None = None          # base64 image (kept for generation context)
