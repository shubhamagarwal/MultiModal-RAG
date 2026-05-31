import time
from openai import OpenAI

from config import API_KEY, BASE_URL, OPENAI_EMBEDDING_MODEL
from logger import get_logger

log = get_logger("embeddings.openai")

_client: OpenAI | None = None
_BATCH_SIZE = 100


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    total = len(texts)
    log.info("Embedding %d text(s) | model=%s | batch_size=%d", total, OPENAI_EMBEDDING_MODEL, _BATCH_SIZE)

    all_embeddings: list[list[float]] = []

    for i in range(0, total, _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        batch = [t if t.strip() else "empty" for t in batch]
        batch_num = i // _BATCH_SIZE + 1
        total_batches = (total + _BATCH_SIZE - 1) // _BATCH_SIZE

        log.info("Embedding batch %d/%d | texts %d-%d",
                 batch_num, total_batches, i + 1, min(i + _BATCH_SIZE, total))

        response = _get_client().embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])
        log.debug("Batch %d/%d complete | dims=%d", batch_num, total_batches,
                  len(response.data[0].embedding) if response.data else 0)

        if i + _BATCH_SIZE < total:
            time.sleep(0.1)

    log.info("Embedding complete | total=%d vectors", len(all_embeddings))
    return all_embeddings
