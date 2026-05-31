"""
RAGAS evaluation for the MultiModal RAG pipeline.

Metrics (no ground truth required):
  - faithfulness       : Is the answer grounded in the retrieved context?
  - answer_relevancy   : Is the answer relevant to the question asked?

Metrics (ground truth required):
  - context_precision  : Of retrieved chunks, what fraction are genuinely relevant?
  - context_recall     : Were all chunks needed to answer the question retrieved?
"""

from __future__ import annotations

from typing import TypedDict

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.metrics import context_precision, context_recall

from config import API_KEY, BASE_URL, OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_MODEL
from logger import get_logger

log = get_logger("evaluation.ragas")


class RAGSample(TypedDict, total=False):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


def _build_llm():
    kwargs = dict(model=OPENAI_CHAT_MODEL, api_key=API_KEY, temperature=0)
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return ChatOpenAI(**kwargs)


def _build_embeddings():
    kwargs = dict(model=OPENAI_EMBEDDING_MODEL, api_key=API_KEY)
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAIEmbeddings(**kwargs)


def evaluate_rag(samples: list[RAGSample]) -> dict:
    has_ground_truth = all("ground_truth" in s for s in samples)
    metrics = [faithfulness, answer_relevancy]
    if has_ground_truth:
        metrics += [context_precision, context_recall]

    log.info("RAGAS evaluation started | samples=%d | metrics=%s",
             len(samples), [m.name for m in metrics])

    for i, s in enumerate(samples):
        log.debug("Sample %d | question='%s...' | contexts=%d | answer_len=%d",
                  i + 1, s["question"][:60], len(s.get("contexts", [])), len(s.get("answer", "")))

    data = {
        "question": [s["question"] for s in samples],
        "answer":   [s["answer"]   for s in samples],
        "contexts": [s["contexts"] for s in samples],
    }
    if has_ground_truth:
        data["ground_truth"] = [s["ground_truth"] for s in samples]

    dataset = Dataset.from_dict(data)

    log.info("Running RAGAS evaluate | model=%s", OPENAI_CHAT_MODEL)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=_build_llm(),
        embeddings=_build_embeddings(),
    )

    scores = dict(result)
    log.info("RAGAS evaluation complete | scores=%s",
             {k: f"{v:.4f}" for k, v in scores.items() if isinstance(v, float)})

    return scores
