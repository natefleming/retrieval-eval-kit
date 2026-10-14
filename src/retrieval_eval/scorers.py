"""Scorers: deterministic IR metrics against ground-truth doc_uris, plus MLflow's built-in retrieval judges.

All scorers are registered to the experiment, so they appear on its Scorers tab and can be reused from the UI (e.g. for
production monitoring). MLflow registers a custom `@scorer` by serializing its function body, so `retrieval_metrics` must
be self-contained: imports inside the body, no closures, no module-level helpers.
"""

from __future__ import annotations

from typing import Any

from mlflow.entities import Feedback, Trace
from mlflow.genai.scorers import RetrievalRelevance, RetrievalSufficiency, Scorer, list_scorers, scorer


@scorer(name="retrieval_metrics")
def retrieval_metrics(trace: Trace, expectations: dict[str, Any] | None = None) -> list[Feedback]:
    """Document-level recall/precision/hit-rate/nDCG at 1, 5 and 10, MRR and latency, from the top-level RETRIEVER span.

    Like MLflow's retrieval judges, only RETRIEVER spans with no RETRIEVER ancestor count, so spans emitted inside a
    wrapped LangChain retriever or tool are ignored in favour of the evaluator's normalized one.

    Retrieved chunks collapse to their `doc_uri` (first occurrence wins), so indexes with different chunking compare
    fairly. Ground truth is `expectations.expected_retrieved_context[*].doc_uri`; without it only latency is reported.
    """
    import math

    from mlflow.entities import Feedback, Span, SpanType

    by_id: dict[str, Span] = {s.span_id: s for s in trace.data.spans}

    def is_top_level(span: Span) -> bool:
        parent: Span | None = by_id.get(span.parent_id or "")
        while parent is not None:
            if parent.span_type == SpanType.RETRIEVER:
                return False
            parent = by_id.get(parent.parent_id or "")
        return True

    spans: list[Span] = [s for s in trace.data.spans if s.span_type == SpanType.RETRIEVER and is_top_level(s)]
    documents: list[dict[str, Any]] = (spans[-1].outputs or []) if spans else []
    retrieved: list[str] = list(dict.fromkeys(
        str(d["metadata"]["doc_uri"]) for d in documents if (d.get("metadata") or {}).get("doc_uri") is not None
    ))
    relevant: set[str] = {str(c["doc_uri"]) for c in (expectations or {}).get("expected_retrieved_context") or []}

    feedbacks: list[Feedback] = [Feedback(name="latency_ms", value=float(trace.info.execution_duration or 0))]
    if not relevant:
        return feedbacks
    hits: list[bool] = [uri in relevant for uri in retrieved]
    first_hit: int | None = next((i for i, hit in enumerate(hits) if hit), None)
    feedbacks.append(Feedback(name="mrr", value=1 / (first_hit + 1) if first_hit is not None else 0.0))
    for k in (1, 5, 10):
        top: list[bool] = hits[:k]
        dcg: float = sum(1 / math.log2(i + 2) for i, hit in enumerate(top) if hit)
        ideal: float = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        feedbacks += [
            Feedback(name=f"recall_at_{k}", value=sum(top) / len(relevant)),
            Feedback(name=f"precision_at_{k}", value=sum(top) / k),
            Feedback(name=f"hit_rate_at_{k}", value=float(any(top))),
            Feedback(name=f"ndcg_at_{k}", value=dcg / ideal),
        ]
    return feedbacks


def make_scorers(judge_model: str, use_llm_judges: bool) -> list[Scorer]:
    scorers: list[Scorer] = [retrieval_metrics]
    if use_llm_judges:
        scorers += [RetrievalRelevance(model=judge_model), RetrievalSufficiency(model=judge_model)]
    return scorers


def _definition(s: Scorer) -> tuple[Any, ...]:
    """What makes two scorers the same: built-in class + judge model + instructions, or a code scorer's source."""
    dump: dict[str, Any] = s.model_dump()
    judge: dict[str, Any] = dump.get("builtin_scorer_pydantic_data") or {}
    return dump.get("builtin_scorer_class"), judge.get("model"), judge.get("instructions"), dump.get("call_source")


def register_scorers(scorers: list[Scorer], experiment_id: str) -> None:
    """Register scorers that are missing from the experiment or whose definition changed (e.g. a new judge_model).

    Unchanged scorers are left alone; registering again would add a new version on every run.
    """
    registered: dict[str, Scorer] = {s.name: s for s in list_scorers(experiment_id=experiment_id)}
    for s in scorers:
        if s.name not in registered or _definition(registered[s.name]) != _definition(s):
            s.register(name=s.name, experiment_id=experiment_id)
            print(f"Registered scorer {s.name}")
