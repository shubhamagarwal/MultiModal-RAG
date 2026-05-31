from config import MAX_RETRIEVAL_RESULTS, MIN_RELEVANCE_SCORE
from logger import get_logger
from src.vectorstore import ChromaStore

log = get_logger("retrieval.retriever")

# One ChromaStore instance per tenant, reused across requests in the same process
_stores: dict[str, ChromaStore] = {}


def get_store(tenant_id: str = "default") -> ChromaStore:
    if tenant_id not in _stores:
        _stores[tenant_id] = ChromaStore(tenant_id=tenant_id)
    return _stores[tenant_id]


def retrieve(query: str, tenant_id: str = "default", n_results: int = MAX_RETRIEVAL_RESULTS) -> list[dict]:
    log.info("Retrieve | tenant=%s | query='%s...' | n_results=%d | threshold=%.2f",
             tenant_id, query[:60], n_results, MIN_RELEVANCE_SCORE)

    raw = get_store(tenant_id).query(query, n_results=n_results * 2)
    log.info("Raw candidates: %d", len(raw))

    filtered = [r for r in raw if r["score"] >= MIN_RELEVANCE_SCORE]
    dropped = len(raw) - len(filtered)
    if dropped:
        log.info("Filtered out %d chunk(s) below score threshold %.2f", dropped, MIN_RELEVANCE_SCORE)

    result = sorted(filtered, key=lambda r: r["score"], reverse=True)[:n_results]
    log.info("Returning %d chunk(s) | scores: %s",
             len(result), [f"{r['score']:.3f}" for r in result])

    for r in result:
        log.info("  -> source=%-30s page=%-4s type=%-6s score=%.4f",
                 r["metadata"].get("source"), r["metadata"].get("page"),
                 r["metadata"].get("chunk_type"), r["score"])

    return result
