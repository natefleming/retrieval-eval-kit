import math
from collections.abc import Callable

import pytest
from mlflow.entities import Feedback, Trace
from mlflow.genai.scorers import RetrievalRelevance, Scorer

from retrieval_eval.scorers import _definition, retrieval_metrics

MakeTrace = Callable[..., Trace]


def run(scorer: Scorer, trace: Trace, relevant: list[str] | None) -> dict[str, float]:
    expectations = None if relevant is None else {"expected_retrieved_context": [{"doc_uri": u} for u in relevant]}
    feedbacks = scorer(trace=trace, expectations=expectations)
    assert isinstance(feedbacks, list) and all(isinstance(f, Feedback) for f in feedbacks)
    return {f.name: float(f.value) for f in feedbacks if isinstance(f.value, (int, float))}


def score(trace: Trace, relevant: list[str] | None) -> dict[str, float]:
    return run(retrieval_metrics, trace, relevant)


def test_hit_at_rank_one(make_trace: MakeTrace) -> None:
    m = score(make_trace(["a", "b", "c"]), ["a"])
    assert m["recall_at_1"] == m["hit_rate_at_1"] == m["ndcg_at_1"] == m["mrr"] == 1.0
    assert m["precision_at_5"] == pytest.approx(1 / 5)
    assert m["latency_ms"] >= 0


def test_hit_at_rank_three(make_trace: MakeTrace) -> None:
    m = score(make_trace(["x", "y", "a"]), ["a"])
    assert m["recall_at_1"] == 0.0 and m["recall_at_5"] == 1.0
    assert m["mrr"] == pytest.approx(1 / 3)
    assert m["ndcg_at_5"] == pytest.approx(1 / math.log2(4))


def test_miss_beyond_ten(make_trace: MakeTrace) -> None:
    m = score(make_trace([str(i) for i in range(10)] + ["a"]), ["a"])
    assert m["recall_at_10"] == m["hit_rate_at_10"] == m["ndcg_at_10"] == 0.0
    assert m["mrr"] == pytest.approx(1 / 11)  # MRR covers every retrieved result, not just the top 10


def test_chunks_collapse_to_documents(make_trace: MakeTrace) -> None:
    # Two chunks of doc "x" occupy one rank, so "a" is at rank 2 once duplicates are removed.
    assert score(make_trace(["x", "x", "a"]), ["a"])["mrr"] == 0.5


def test_multiple_relevant(make_trace: MakeTrace) -> None:
    m = score(make_trace(["a", "x", "b"]), ["a", "b"])
    assert m["recall_at_5"] == 1.0 and m["recall_at_1"] == 0.5
    assert m["ndcg_at_5"] == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))


def test_only_top_level_retriever_span_counts(make_trace: MakeTrace) -> None:
    """A wrapped retriever's own (inner) RETRIEVER span must not replace the evaluator's normalized one."""
    m = score(make_trace(["a"], nested=["x", "y"]), ["a"])
    assert m["mrr"] == 1.0


def test_no_ground_truth_reports_latency_only(make_trace: MakeTrace) -> None:
    assert list(score(make_trace(["a"]), None)) == ["latency_ms"]


def test_survives_registration_round_trip(make_trace: MakeTrace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Registered scorers are rebuilt from their serialized source, so the body must not depend on module state.

    MLflow treats any process that has imported uvicorn (databricks-langchain does) as a tracking server and refuses to
    rebuild scorer code there, so pin client behaviour for this test.
    """
    monkeypatch.setattr("mlflow.genai.scorers.base._is_tracking_server_process", lambda: False)
    rebuilt: Scorer = Scorer.model_validate(retrieval_metrics.model_dump())
    assert run(rebuilt, make_trace(["x", "a"]), ["a"])["mrr"] == 0.5


def test_definition_works_on_non_executing_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    """register_scorers compares definitions of scorers read back from the registry, which MLflow may load as
    non-executing metadata handles (see above); the comparison must still work."""
    monkeypatch.setattr("mlflow.genai.scorers.base._is_tracking_server_process", lambda: True)
    handle: Scorer = Scorer.model_validate(retrieval_metrics.model_dump())
    assert _definition(handle) == _definition(retrieval_metrics)


def test_definition_detects_judge_model_change() -> None:
    a, b = RetrievalRelevance(model="databricks:/m1"), RetrievalRelevance(model="databricks:/m1")
    assert _definition(a) == _definition(b)
    assert _definition(a) != _definition(RetrievalRelevance(model="databricks:/m2"))
    assert _definition(retrieval_metrics) == _definition(Scorer.model_validate(retrieval_metrics.model_dump()))
