from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from logger import get_logger
from src.ingestion.base import DocumentChunk

log = get_logger("processing.chunker")

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def split_text_chunk(chunk: DocumentChunk) -> list[DocumentChunk]:
    if len(chunk.content) <= CHUNK_SIZE:
        log.debug("Chunk fits in one piece | source=%s page=%s | %d chars",
                  chunk.source, chunk.page, len(chunk.content))
        return [chunk]

    parts = _splitter.split_text(chunk.content)
    log.debug("Split chunk | source=%s page=%s | %d chars -> %d part(s)",
              chunk.source, chunk.page, len(chunk.content), len(parts))

    return [
        DocumentChunk(
            content=part,
            chunk_type=chunk.chunk_type,
            source=chunk.source,
            page=chunk.page,
            metadata={**chunk.metadata, "split_index": i},
        )
        for i, part in enumerate(parts)
    ]
