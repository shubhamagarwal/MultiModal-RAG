import warnings
from openai import OpenAI

from config import VISION_API_KEY, VISION_BASE_URL, VISION_ENABLED, OPENAI_VISION_MODEL
from logger import get_logger
from src.ingestion.base import DocumentChunk

log = get_logger("processing.image_processor")

_client: OpenAI | None = None

VISION_PROMPT = (
    "Describe this image in detail for a retrieval-augmented generation system. "
    "Include: all visible text, charts/graphs data, key visual elements, colors, "
    "layout, and any information a user might query about. Be thorough and precise."
)

_REFUSAL_MARKERS = [
    "i cannot process",
    "i can't process",
    "unable to process",
    "cannot view",
    "i'm sorry",
    "i am sorry",
    "no image",
    "don't see any image",
]


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
    return _client


def _is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _fallback_description(chunk: DocumentChunk) -> str:
    meta_parts = [f"{k}: {v}" for k, v in chunk.metadata.items()]
    meta_str = ", ".join(meta_parts) if meta_parts else "no additional metadata"
    return (
        f"[Image from {chunk.source}, page {chunk.page}] "
        f"({meta_str}) — image description unavailable: "
        "set OPENAI_API_KEY alongside GITHUB_TOKEN to enable GPT-4o Vision."
    )


def describe_image(chunk: DocumentChunk) -> DocumentChunk:
    if not chunk.image_b64:
        log.warning("describe_image called but chunk has no image_b64 | source=%s", chunk.source)
        return chunk

    if not VISION_ENABLED:
        log.warning(
            "Vision disabled (OPENAI_API_KEY not set) | source=%s page=%s — storing placeholder",
            chunk.source, chunk.page,
        )
        chunk.content = _fallback_description(chunk)
        return chunk

    log.info("Calling GPT-4o Vision | source=%s page=%s model=%s",
             chunk.source, chunk.page, OPENAI_VISION_MODEL)
    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{chunk.image_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )

        description = response.choices[0].message.content or ""

        if _is_refusal(description):
            log.warning("Vision model refused to describe image | source=%s page=%s — using fallback",
                        chunk.source, chunk.page)
            chunk.content = _fallback_description(chunk)
        else:
            log.info("Vision description generated | source=%s page=%s | %d chars",
                     chunk.source, chunk.page, len(description))
            chunk.content = f"[Image from {chunk.source}, page {chunk.page}]\n{description}"

    except Exception as e:
        log.error("Vision API error | source=%s page=%s | %s", chunk.source, chunk.page, e)
        chunk.content = (
            f"[Image from {chunk.source}, page {chunk.page}] "
            f"— vision processing failed: {e}"
        )

    return chunk
