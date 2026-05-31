"""
PII guardrail using Microsoft Presidio.

Two modes:
  - redact(text)      : replace PII with <ENTITY_TYPE> placeholders — used during ingestion
  - check_query(text) : raise PIIBlockedError if the query contains PII — used at query time
"""

from __future__ import annotations
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from logger import get_logger

log = get_logger("security.pii_guard")

WATCHED_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_ITIN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IBAN_CODE",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
    "NRP",
]

CONFIDENCE_THRESHOLD = 0.7

_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def _get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        log.info("┌─ [PII GUARD] Initialising Presidio AnalyzerEngine (spaCy en_core_web_lg) ─┐")
        _analyzer = AnalyzerEngine()
        log.info("└─ [PII GUARD] Presidio ready | watching %d entity types | threshold=%.2f ──┘",
                 len(WATCHED_ENTITIES), CONFIDENCE_THRESHOLD)
    return _analyzer


def _get_anonymizer() -> AnonymizerEngine:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _anonymizer


@dataclass
class PIIFinding:
    entity_type: str
    score: float
    start: int
    end: int


class PIIBlockedError(ValueError):
    def __init__(self, findings: list[PIIFinding]):
        self.findings = findings
        types = sorted({f.entity_type for f in findings})
        super().__init__(
            f"Query blocked: contains PII — detected types: {', '.join(types)}. "
            "Please remove personal information before querying."
        )


def detect(text: str) -> list[PIIFinding]:
    if not text.strip():
        return []

    results = _get_analyzer().analyze(
        text=text,
        entities=WATCHED_ENTITIES,
        language="en",
        score_threshold=CONFIDENCE_THRESHOLD,
    )
    return [PIIFinding(r.entity_type, r.score, r.start, r.end) for r in results]


def redact(text: str) -> tuple[str, list[PIIFinding]]:
    if not text.strip():
        return text, []

    analyzer_results = _get_analyzer().analyze(
        text=text,
        entities=WATCHED_ENTITIES,
        language="en",
        score_threshold=CONFIDENCE_THRESHOLD,
    )

    if not analyzer_results:
        return text, []

    findings = [PIIFinding(r.entity_type, r.score, r.start, r.end) for r in analyzer_results]

    redacted = _get_anonymizer().anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in WATCHED_ENTITIES
        },
    )

    log.info("  [PII GUARD] Redacted %d item(s):", len(findings))
    for f in sorted(findings, key=lambda x: x.start):
        original_span = text[f.start:f.end]
        masked = original_span[0] + "*" * (len(original_span) - 2) + original_span[-1] \
            if len(original_span) > 2 else "***"
        log.info("    %-22s | confidence=%.2f | value='%s'", f.entity_type, f.score, masked)

    return redacted.text, findings


def check_query(text: str) -> None:
    findings = detect(text)
    if not findings:
        log.info("  [PII GUARD] Query scan passed — no PII detected")
        return

    log.info("  [PII GUARD] Query BLOCKED | %d PII item(s) found:", len(findings))
    for f in sorted(findings, key=lambda x: x.start):
        original_span = text[f.start:f.end]
        masked = original_span[0] + "*" * (len(original_span) - 2) + original_span[-1] \
            if len(original_span) > 2 else "***"
        log.info("    %-22s | confidence=%.2f | value='%s'", f.entity_type, f.score, masked)

    raise PIIBlockedError(findings)
