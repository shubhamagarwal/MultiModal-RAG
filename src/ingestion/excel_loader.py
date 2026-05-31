from pathlib import Path

import pandas as pd

from logger import get_logger
from .base import DocumentChunk

log = get_logger("ingestion.excel")


def load_excel(path: str) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    source = Path(path).name
    ext = Path(path).suffix.lower()
    log.info("Opening %s: %s", ext.lstrip(".").upper(), source)

    if ext == ".csv":
        sheets = {"Sheet1": pd.read_csv(path)}
        log.info("CSV loaded as single sheet")
    else:
        sheets = pd.read_excel(path, sheet_name=None)
        log.info("Excel workbook has %d sheet(s): %s", len(sheets), list(sheets.keys()))

    for sheet_name, df in sheets.items():
        original_rows = len(df)
        df = df.dropna(how="all").fillna("")
        log.info("Sheet '%s': %d row(s) x %d col(s) (dropped %d empty rows)",
                 sheet_name, len(df), len(df.columns), original_rows - len(df))

        if df.empty:
            log.warning("Sheet '%s' is empty after cleaning — skipping", sheet_name)
            continue

        md_table = df.to_markdown(index=False)
        chunks.append(DocumentChunk(
            content=md_table,
            chunk_type="table",
            source=source,
            page=1,
            metadata={"file_type": ext.lstrip("."), "sheet": sheet_name},
        ))
        log.info("Sheet '%s': converted to markdown table (%d chars)", sheet_name, len(md_table))

    log.info("Excel load complete | chunks=%d", len(chunks))
    return chunks
