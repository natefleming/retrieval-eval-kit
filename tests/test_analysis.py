import mlflow
import pandas as pd
import plotly.graph_objects as go
import pytest
from mlflow.entities import Document, SpanType

from retrieval_eval.analysis import (
    chart_html, fmt, fmt_change, key_findings, metrics_table, quality_latency_chart, recall_curve_chart, ranking_chart, recommend,
    recommendation_html, results_table,
)
from retrieval_eval.config import ExperimentConfig
from retrieval_eval.config import Settings, Variant
from retrieval_eval.evaluate import per_question_variance
from retrieval_eval.scorers import retrieval_metrics

N = 100


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"source_index": "c.s.i", "doc_uri_column": "id", "content_columns": ["t"],
                               "output_schema": "c.o", "agent_description": "test", "baseline_variant": "base"}
    return Settings.model_validate({**base, **overrides})


def variants(*names: str) -> list[Variant]:
    return [Variant(name=n, retriever={"factory": "f.py:f"}) for n in names]


def summary(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Shape of evaluate.run output: '<metric>/mean' columns (variance omitted → Bernoulli estimate)."""
    df = pd.DataFrame({name: {f"{k}/mean": v for k, v in metrics.items()} for name, metrics in rows.items()}).T
    df["run_id"] = [f"run-{n}" for n in df.index]
    return df


def test_clear_winner_is_recommended() -> None:
    s = summary({
        "base": {"recall_at_10": 0.60, "ndcg_at_10": 0.50, "latency_ms": 100},
        "better": {"recall_at_10": 0.90, "ndcg_at_10": 0.70, "latency_ms": 120},
    })
    rec = recommend(metrics_table(s, N), settings(), variants("base", "better"), N)
    assert rec.variant == "better" and rec.tied == ["better"] and rec.clear_improvement
    assert rec.change == pytest.approx(0.30)


def test_ties_are_broken_by_secondary_metric_then_latency() -> None:
    s = summary({
        "base": {"recall_at_10": 0.60, "ndcg_at_10": 0.50, "latency_ms": 100},
        "leader": {"recall_at_10": 0.91, "ndcg_at_10": 0.60, "latency_ms": 900},
        "close": {"recall_at_10": 0.89, "ndcg_at_10": 0.75, "latency_ms": 300},
    })
    rec = recommend(metrics_table(s, N), settings(), variants("base", "leader", "close"), N)
    assert set(rec.tied) == {"leader", "close"} and rec.variant == "close" and rec.leader == "leader"


def test_latency_budget_excludes_slow_variants() -> None:
    s = summary({
        "base": {"recall_at_10": 0.60, "latency_ms": 100},
        "slow": {"recall_at_10": 0.95, "latency_ms": 2000},
    })
    rec = recommend(metrics_table(s, N), settings(latency_budget_ms=500), variants("base", "slow"), N)
    assert rec.variant == "base" and rec.excluded_by_latency == ["slow"]


def test_unknown_primary_metric_is_a_clear_error() -> None:
    s = summary({"base": {"recall_at_10": 0.6}})
    with pytest.raises(ValueError, match="primary_metric"):
        recommend(metrics_table(s, N), settings(primary_metric="nope"), variants("base"), N)


def test_presentation_builds() -> None:
    metrics = {"recall_at_1": 0.5, "recall_at_5": 0.7, "recall_at_10": 0.8, "ndcg_at_10": 0.6, "mrr": 0.55, "latency_ms": 300}
    s = summary({"base": metrics, "other": {k: v * 0.9 for k, v in metrics.items()}, "best": {k: min(v * 1.1, 1) for k, v in metrics.items()}})
    table = metrics_table(s, N)
    rec = recommend(table, settings(), variants("base", "other", "best"), N)
    for fig in (ranking_chart(table, rec), recall_curve_chart(table, rec), quality_latency_chart(table, rec)):
        assert isinstance(fig, go.Figure) and fig.data
        assert "cdn.plot.ly" in chart_html(fig)
    config = ExperimentConfig(settings=settings(), variants=variants("base", "other", "best"))
    assert "★ best" in results_table(table, rec, config).to_html() and "Found in top 10 (recall@10)" in results_table(table, rec, config).to_html()
    html = recommendation_html(rec, table, variants("best")[0], {"embedding_model": "databricks-gte-large-en", "retriever.query_type": "HYBRID"})
    assert "Recommended configuration" in html and "databricks-gte-large-en" in html and "Hybrid (meaning + keywords)" in html
    assert any("Best at finding" in f for f in key_findings(table, rec))


def test_plain_language_formatting() -> None:
    assert fmt("recall_at_10", 0.95) == "95%" and fmt("ndcg_at_10", 0.8134) == "0.81" and fmt("latency_ms", 361.5) == "362 ms"
    assert fmt_change("recall_at_10", 0.067) == "+6.7 pts" and fmt_change("latency_ms", -120.4) == "-120 ms"
    assert fmt_change("recall_at_10", 0.067, long=True) == "+6.7 percentage points"


def test_evaluate_reports_variance_for_margins_of_error() -> None:
    """End to end on a local store: per-question variance from the evaluation results feeds the margins of error."""

    @mlflow.trace(name="retrieve", span_type=SpanType.RETRIEVER)
    def retrieve(query: str) -> list[Document]:
        return [Document(page_content=query, metadata={"doc_uri": "a"})]

    data = [{"inputs": {"query": q}, "expectations": {"expected_retrieved_context": [{"doc_uri": d}]}} for q, d in
            [("q1", "a"), ("q2", "b"), ("q3", "a"), ("q4", "b")]]
    result = mlflow.genai.evaluate(data=data, predict_fn=retrieve, scorers=[retrieval_metrics])
    variances = per_question_variance(result.result_df)
    assert result.metrics["recall_at_1/mean"] == pytest.approx(0.5)
    assert variances["recall_at_1/variance"] == pytest.approx(1 / 3)  # values 1,0,1,0 → sample variance 1/3
    table = metrics_table(pd.DataFrame([{**result.metrics, **variances}], index=["v"]), 4)
    assert table.at["v", "recall_at_1_se"] == pytest.approx((1 / 3 / 4) ** 0.5)


def test_search_time_is_preferred_and_big_slowdowns_are_flagged() -> None:
    s = summary({
        "base": {"recall_at_10": 0.60, "ndcg_at_10": 0.5, "search_time_ms": 400, "latency_ms": 900},
        "slow_but_better": {"recall_at_10": 0.95, "ndcg_at_10": 0.8, "search_time_ms": 1600, "latency_ms": 40000},
    })
    table = metrics_table(s, N)
    rec = recommend(table, settings(latency_budget_ms=1000), variants("base", "slow_but_better"), N)
    assert rec.variant == "base"  # the budget applies to one-at-a-time search time (1600 ms), not latency under load
    rec = recommend(table, settings(), variants("base", "slow_but_better"), N)
    html = recommendation_html(rec, table, variants("slow_but_better")[0], {})
    assert "about 4× slower than today" in html and "1600 ms" in html and "40000" not in html


def test_identical_scores_are_described_as_the_same() -> None:
    s = summary({
        "base": {"recall_at_10": 0.50, "ndcg_at_10": 0.40},
        "first": {"recall_at_10": 0.90, "ndcg_at_10": 0.60},
        "second": {"recall_at_10": 0.90, "ndcg_at_10": 0.80},
    })
    table = metrics_table(s, N)
    rec = recommend(table, settings(), variants("base", "first", "second"), N)
    assert rec.variant == "second" and rec.leader == "first"
    assert "first scored the same on found in top 10" in recommendation_html(rec, table, variants("second")[0], {})


def test_ranking_advice_depends_on_whether_the_variant_already_reranks() -> None:
    s = summary({"base": {"recall_at_10": 0.6, "recall_at_1": 0.3}, "best": {"recall_at_10": 0.95, "recall_at_1": 0.6}})
    table = metrics_table(s, N)
    rec = recommend(table, settings(), variants("base", "best"), N)
    gap = lambda reranked: next(f for f in key_findings(table, rec, reranked) if "Finding vs. ranking" in f)
    assert "a reranker or a different search mode" in gap(set())
    assert "a different or stronger reranker" in gap({"best"})


@pytest.mark.parametrize(("better", "label"), [(0.95, "High."), (0.73, "Moderate."), (0.66, "Low to moderate.")])
def test_confidence_wording_is_tiered(better: float, label: str) -> None:
    # "new" also ranks better (ndcg), so it wins a statistical tie on recall too (the low-confidence case).
    s = summary({"base": {"recall_at_10": 0.60, "ndcg_at_10": 0.5}, "new": {"recall_at_10": better, "ndcg_at_10": 0.7}})
    table = metrics_table(s, N)
    rec = recommend(table, settings(), variants("base", "new"), N)
    assert rec.variant == "new" and label in recommendation_html(rec, table, variants("new")[0], {})
