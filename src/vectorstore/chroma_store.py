import hashlib

import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR
from logger import get_logger
from src.ingestion.base import DocumentChunk
from src.embeddings import embed_texts

log = get_logger("vectorstore.chroma")

_DEFAULT_TENANT = "default"


def _collection_name(tenant_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tenant_id).lower()
    return f"rag_{safe}"


class ChromaStore:
    def __init__(self, tenant_id: str = _DEFAULT_TENANT):
        collection = _collection_name(tenant_id)
        log.info("Initialising ChromaDB | path=%s | tenant=%s | collection=%s",
                 CHROMA_PERSIST_DIR, tenant_id, collection)
        self._client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.tenant_id = tenant_id
        log.info("ChromaDB ready | tenant=%s | existing chunks=%d", tenant_id, self._collection.count())

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            log.warning("add_chunks called with empty list — nothing to store")
            return 0

        log.info("Storing %d chunk(s) into ChromaDB", len(chunks))
        texts = [c.content for c in chunks]
        embeddings = embed_texts(texts)

        seen_ids: set[str] = set()
        ids, documents, metadatas, embeds = [], [], [], []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = hashlib.md5(chunk.content.encode()).hexdigest()

            if chunk_id in seen_ids:
                log.info("Skipping duplicate chunk | source=%s page=%s", chunk.source, chunk.page)
                continue
            seen_ids.add(chunk_id)

            meta = {
                "source": chunk.source,
                "page": chunk.page,
                "chunk_type": chunk.chunk_type,
                **{k: str(v) for k, v in chunk.metadata.items()},
            }
            if chunk.image_b64:
                meta["has_image"] = "true"

            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append(meta)
            embeds.append(embedding)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeds,
        )
        log.info("Upsert complete | stored=%d | total_in_db=%d", len(ids), self._collection.count())
        return len(ids)

    def query(self, query_text: str, n_results: int = 6) -> list[dict]:
        log.info("Querying ChromaDB | n_results=%d | query='%s...'",
                 n_results, query_text[:60])
        embedding = embed_texts([query_text])[0]
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1 - dist
            output.append({"content": doc, "metadata": meta, "score": score})
            log.debug("  candidate | source=%s page=%s type=%s score=%.4f",
                      meta.get("source"), meta.get("page"), meta.get("chunk_type"), score)

        log.info("Query returned %d candidate(s)", len(output))
        return output

    def delete_source(self, source: str) -> None:
        log.info("Deleting all chunks for source: %s", source)
        results = self._collection.get(where={"source": source})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            log.info("Deleted %d chunk(s) for source '%s'", len(results["ids"]), source)
        else:
            log.warning("No chunks found for source '%s'", source)

    def list_sources(self) -> list[str]:
        results = self._collection.get(include=["metadatas"])
        sources = sorted({m["source"] for m in results["metadatas"] if m})
        log.debug("Listed %d unique source(s)", len(sources))
        return sources

    def count(self) -> int:
        return self._collection.count()
