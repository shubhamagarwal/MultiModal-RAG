import re
from collections.abc import Generator
from openai import OpenAI

from config import API_KEY, BASE_URL, OPENAI_CHAT_MODEL
from logger import get_logger

log = get_logger("generation.generator")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


SYSTEM_PROMPT = """You are a precise assistant that answers questions strictly from the provided context chunks.

Rules:
- Each context chunk is labelled [1], [2], [3], etc.
- Cite every claim using its chunk number, e.g. [1] or [2,3].
- If the answer is not present in any chunk, respond exactly: "The answer is not available in the provided documents."
- Do NOT use knowledge outside the provided context.
- Format your response in clear, readable markdown."""


def build_context(retrieved: list[dict]) -> str:
    parts = []
    for i, item in enumerate(retrieved, 1):
        meta = item["metadata"]
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        chunk_type = meta.get("chunk_type", "text")
        score = item.get("score", 0)
        header = f"[{i}] {source} | page {page} | type: {chunk_type} | score: {score:.2f}"
        parts.append(f"{header}\n{item['content']}")
    return "\n\n---\n\n".join(parts)


def _extract_cited_indices(answer: str) -> set[int]:
    cited = set()
    for match in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', answer):
        for num in match.group(1).split(","):
            cited.add(int(num.strip()))
    return cited


def _build_messages(
    query: str,
    retrieved: list[dict],
    history: list[dict] | None,
) -> list[dict]:
    context = build_context(retrieved)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    return messages


def _resolve_citations(answer: str, retrieved: list[dict]) -> list[dict]:
    cited_indices = _extract_cited_indices(answer)
    cited_chunks = [
        retrieved[i - 1]
        for i in sorted(cited_indices)
        if 1 <= i <= len(retrieved)
    ]
    log.info("Citations in answer: %s | cited_chunks=%d", sorted(cited_indices), len(cited_chunks))
    for c in cited_chunks:
        log.info("  cited -> source=%s page=%s", c["metadata"].get("source"), c["metadata"].get("page"))
    return cited_chunks


def generate_answer(
    query: str,
    retrieved: list[dict],
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    if not retrieved:
        log.warning("generate_answer called with empty retrieved list")
        return "No relevant context found in the knowledge base.", []

    log.info("Generating answer | model=%s | context_chunks=%d | query='%s...'",
             OPENAI_CHAT_MODEL, len(retrieved), query[:60])

    messages = _build_messages(query, retrieved, history)
    response = _get_client().chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
    )

    answer = response.choices[0].message.content or ""
    usage = response.usage
    log.info("LLM response | prompt_tokens=%s completion_tokens=%s total_tokens=%s",
             usage.prompt_tokens if usage else "?",
             usage.completion_tokens if usage else "?",
             usage.total_tokens if usage else "?")

    return answer, _resolve_citations(answer, retrieved)


def stream_answer(
    query: str,
    retrieved: list[dict],
    history: list[dict] | None = None,
) -> Generator[str, None, list[dict]]:
    """Yield text delta tokens from the LLM stream.

    Usage::
        gen = stream_answer(query, retrieved, history)
        full_text = ""
        try:
            while True:
                token = next(gen)
                full_text += token
        except StopIteration as exc:
            cited_chunks = exc.value
    """
    if not retrieved:
        log.warning("stream_answer called with empty retrieved list")
        yield "No relevant context found in the knowledge base."
        return []

    log.info("Streaming answer | model=%s | context_chunks=%d | query='%s...'",
             OPENAI_CHAT_MODEL, len(retrieved), query[:60])

    messages = _build_messages(query, retrieved, history)
    stream = _get_client().chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
        stream=True,
    )

    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_answer += delta
            yield delta

    log.info("Stream complete | answer_len=%d", len(full_answer))
    return _resolve_citations(full_answer, retrieved)
