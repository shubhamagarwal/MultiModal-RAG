from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from logger import get_logger
from .base import DocumentChunk

log = get_logger("ingestion.web")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MultiModalRAG/1.0)"}


def load_url(url: str) -> list[DocumentChunk]:
    log.info("Fetching URL: %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    log.info("HTTP %d | content-type: %s | size: %d bytes",
             resp.status_code, resp.headers.get("content-type", "?"), len(resp.content))
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    removed = 0
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "meta", "link"]):
        tag.decompose()
        removed += 1
    log.debug("Removed %d boilerplate tag(s)", removed)

    title = soup.title.string.strip() if soup.title else urlparse(url).netloc
    body = soup.get_text(separator="\n", strip=True)
    lines = [l for l in body.splitlines() if l.strip()]
    text = "\n".join(lines)

    log.info("Web load complete | title='%s' | %d lines | %d chars", title, len(lines), len(text))

    return [
        DocumentChunk(
            content=text,
            chunk_type="text",
            source=url,
            page=1,
            metadata={"file_type": "url", "title": title, "url": url},
        )
    ]
