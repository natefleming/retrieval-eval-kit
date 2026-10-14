"""Analysis of a sweep: a results table, key findings, charts, and a recommended configuration.

Everything here is written for semi-technical readers: metrics carry plain-language names (with the technical name in
brackets), proportions are shown as percentages, and the recommendation explains itself in words.

Decision rule (deliberately simple and explainable):
1. Drop variants slower than `latency_budget_ms` (if set).
2. The leader is the variant with the best mean `primary_metric`.
3. Variants whose primary metric is within the margin of error of the leader (95% interval of the difference, from
   each metric's per-question variance and the number of questions) are treated as tied with it.
4. Among the tied variants, recommend the best `secondary_metric`, then the lowest latency.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any, Literal

import mlflow
import pandas as pd
import plotly.graph_objects as go

from retrieval_eval.config import ExperimentConfig, Settings, Variant

Z_95: float = 1.96
CUTOFFS: tuple[int, ...] = (1, 5, 10)

# Validated palette (dataviz reference instance, light surface): two roles plus neutral context.
SURFACE: str = "#fcfcfb"
TEXT_PRIMARY: str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
GRID: str = "#e6e5e1"
RECOMMENDED: str = "#2a78d6"  # categorical slot 1 (blue)
BASELINE: str = "#eb6834"  # categorical slot 2 (orange)
OTHER: str = "#a3a29c"  # neutral gray for context variants
SEQUENTIAL: tuple[str, ...] = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

Kind = Literal["pct", "score", "ms"]


@dataclass(frozen=True)
class MetricInfo:
    """How a metric is named, explained and formatted for readers."""

    label: str  # plain-language name
    technical: str  # the standard name, shown in brackets
    meaning: str  # one sentence: what the number tells you
    kind: Kind  # pct: share of questions/results (shown as %); score: 0.00–1.00; ms: milliseconds
    higher_is_better: bool = True

    @property
    def title(self) -> str:
        return f"{self.label} ({self.technical})"


METRICS: dict[str, MetricInfo] = {
    "recall_at_1": MetricInfo("Ranked first", "recall@1", "Share of questions where the correct document is the #1 result.", "pct"),
    "recall_at_5": MetricInfo("Found in top 5", "recall@5", "Share of questions where the correct document is in the top 5 results.", "pct"),
    "recall_at_10": MetricInfo("Found in top 10", "recall@10", "Share of questions where the correct document is anywhere in the top 10 results.", "pct"),
    "hit_rate_at_10": MetricInfo("Found in top 10", "hit rate@10", "Share of questions with at least one correct document in the top 10.", "pct"),
    "precision_at_10": MetricInfo("Correct share of top 10", "precision@10", "Share of the top 10 results that are correct (at most 10% when each question has one correct document).", "pct"),
    "ndcg_at_5": MetricInfo("Ranking score, top 5", "nDCG@5", "How high the correct document is ranked within the top 5: 1.00 = always first, 0 = never in the top 5.", "score"),
    "ndcg_at_10": MetricInfo("Ranking score", "nDCG@10", "How high the correct document is ranked within the top 10: 1.00 = always first, 0.50 = typically third, 0 = never in the top 10.", "score"),
    "mrr": MetricInfo("Position score", "MRR", "1 ÷ the position of the correct document, averaged: 1.00 = always #1, 0.50 = typically #2, 0.33 = typically #3.", "score"),
    "retrieval_relevance": MetricInfo("Relevant results", "AI judge: relevance", "Share of returned results that an AI judge considers relevant to the question (it also credits relevant results other than the expected one).", "pct"),
    "retrieval_sufficiency": MetricInfo("Enough to answer", "AI judge: sufficiency", "Share of questions where an AI judge finds that the returned results contain all the facts needed to answer.", "pct"),
    "search_time_ms": MetricInfo("Search time", "one search at a time", "Average time for one search, measured one question at a time after a warm-up (lower is better).", "ms", higher_is_better=False),
    "latency_ms": MetricInfo("Search time under load", "latency", "Average time per search while the evaluation runs many questions in parallel; includes queueing (lower is better).", "ms", higher_is_better=False),
}
TABLE_METRICS: tuple[str, ...] = (
    "recall_at_10", "recall_at_5", "recall_at_1", "ndcg_at_10", "mrr", "retrieval_relevance", "retrieval_sufficiency",
    "search_time_ms",
)
SLOWDOWN_WARNING: float = 2.0  # flag a recommendation that is this many times slower than today's setup
CLEAR_MARGIN: float = 1.5  # "High" confidence needs an improvement this many times the margin of error


def time_metric(table: pd.DataFrame) -> str:
    """The speed measure to report: one-at-a-time search time when measured, else trace latency (under load)."""
    return "search_time_ms" if "search_time_ms" in table else "latency_ms"


def _table_metrics(table: pd.DataFrame) -> list[str]:
    metrics: list[str] = [m for m in TABLE_METRICS if m in table]
    if "search_time_ms" not in table and "latency_ms" in table:
        metrics.append("latency_ms")
    return metrics

SEARCH_MODES: dict[str, str] = {"ANN": "Vector (meaning-based)", "HYBRID": "Hybrid (meaning + keywords)", "FULL_TEXT": "Keyword"}
PARAM_LABELS: dict[str, str] = {
    "retriever": "Retriever", "retriever.query_type": "Search mode", "retriever.rerank_columns": "Reranker uses",
    "retriever.num_results": "Results per search", "retriever.filters": "Filters", "retriever.score_threshold": "Minimum score",
    "index": "Index", "index_source": "Index origin", "embedding_model": "Embedding model", "embedded_text": "Text embedded",
    "chunking": "Chunking", "chunk_size": "Chunk size (characters)", "chunk_overlap": "Chunk overlap (characters)",
    "k": "Default results per search", "text_columns": "Text embedded",  # text_columns: runs logged before v0.1
}
PARAM_ORDER: tuple[str, ...] = (
    "retriever.query_type", "retriever.rerank_columns", "retriever.num_results", "retriever.filters",
    "retriever.score_threshold", "embedding_model", "embedded_text", "text_columns", "chunking", "chunk_size",
    "chunk_overlap", "index_source", "index", "retriever",
)


def info(metric: str) -> MetricInfo:
    return METRICS.get(metric, MetricInfo(metric, metric, "", "score"))


def fmt(metric: str, value: float) -> str:
    """A value as a reader sees it: 95% / 0.81 / 362 ms."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    kind: Kind = info(metric).kind
    return f"{value:.0%}" if kind == "pct" else f"{value:.0f} ms" if kind == "ms" else f"{value:.2f}"


def fmt_change(metric: str, delta: float, long: bool = False) -> str:
    """A difference as a reader sees it: +6.7 pts (or "+6.7 percentage points" in sentences) / +0.05 / -120 ms."""
    kind: Kind = info(metric).kind
    if kind == "pct":
        return f"{delta * 100:+.1f} {'percentage points' if long else 'pts'}"
    return f"{delta:+.0f} ms" if kind == "ms" else f"{delta:+.2f}"


@dataclass(frozen=True)
class Recommendation:
    variant: str  # recommended variant
    baseline: str  # the configuration it is compared against (today's setup)
    leader: str  # best on the primary metric (may differ from `variant` after tie-breaking)
    primary_metric: str
    secondary_metric: str
    tied: list[str]  # variants too close to the leader to tell apart (includes the leader)
    excluded_by_latency: list[str]
    num_questions: int
    change: float  # recommended − baseline on the primary metric
    margin: float  # 95% margin of error of that change
    latency_budget_ms: float | None = None

    @property
    def clear_improvement(self) -> bool:
        return self.variant != self.baseline and self.change > self.margin


def metrics_table(summary: pd.DataFrame, num_questions: int) -> pd.DataFrame:
    """One row per variant: `<metric>` (mean) and `<metric>_se` (standard error) columns, from `evaluate.run` output."""
    table: pd.DataFrame = pd.DataFrame(index=summary.index)
    for column in summary.columns:
        if column.endswith("/mean"):
            name: str = column.removesuffix("/mean")
            table[name] = summary[column].astype(float)
            variance_column: str = f"{name}/variance"
            if variance_column in summary:
                variance: pd.Series = summary[variance_column].astype(float)
            else:  # binary metrics (recall/hit rate with one relevant document): Bernoulli variance
                variance = table[name] * (1 - table[name])
            table[f"{name}_se"] = (variance.clip(lower=0) / max(num_questions, 1)).pow(0.5)
    return table


def _value(table: pd.DataFrame, variant: str, metric: str) -> float:
    return float(table.at[variant, metric]) if metric in table and variant in table.index else float("nan")


def _margin(table: pd.DataFrame, metric: str, a: str, b: str) -> float:
    """95% margin of error of the difference between two variants on `metric`."""
    se: list[float] = [_value(table, n, f"{metric}_se") for n in (a, b)]
    return Z_95 * math.sqrt(sum((0.0 if math.isnan(s) else s) ** 2 for s in se))


def recommend(table: pd.DataFrame, settings: Settings, variants: list[Variant], num_questions: int) -> Recommendation:
    primary, secondary = settings.primary_metric, settings.secondary_metric
    if primary not in table:
        raise ValueError(f"primary_metric {primary!r} was not computed; available: {sorted(c for c in table if not c.endswith('_se'))}")
    names: list[str] = [v.name for v in variants if v.name in table.index]
    baseline: str = settings.baseline_variant if settings.baseline_variant in names else names[0]

    eligible: list[str] = names
    excluded: list[str] = []
    speed: str = time_metric(table)
    if settings.latency_budget_ms is not None and speed in table:
        eligible = [n for n in names if _value(table, n, speed) <= settings.latency_budget_ms]
        excluded = [n for n in names if n not in eligible]
        if not eligible:
            raise ValueError(f"no variant meets latency_budget_ms={settings.latency_budget_ms}")

    leader: str = max(eligible, key=lambda n: _value(table, n, primary))
    tied: list[str] = [
        n for n in eligible if _value(table, leader, primary) - _value(table, n, primary) <= _margin(table, primary, leader, n)
    ]

    def tie_break(n: str) -> tuple[float, float]:
        sec: float = _value(table, n, secondary)
        lat: float = _value(table, n, speed)
        return (sec if not math.isnan(sec) else -math.inf, -(lat if not math.isnan(lat) else 0))

    chosen: str = max(tied, key=tie_break)
    return Recommendation(
        variant=chosen, baseline=baseline, leader=leader, primary_metric=primary, secondary_metric=secondary, tied=tied,
        excluded_by_latency=excluded, num_questions=num_questions,
        change=_value(table, chosen, primary) - _value(table, baseline, primary),
        margin=_margin(table, primary, chosen, baseline), latency_budget_ms=settings.latency_budget_ms,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Text: results table, key findings, recommendation
# ---------------------------------------------------------------------------------------------------------------------


def _marker(name: str, rec: Recommendation) -> str:
    return "★ " if name == rec.variant else "◆ " if name == rec.baseline else ""


def results_table(table: pd.DataFrame, rec: Recommendation, config: ExperimentConfig) -> Any:
    """One row per variant with a plain description and the key metrics, each column shaded from worst (light) to best
    (dark blue). ★ = recommended, ◆ = today's setup (baseline)."""
    descriptions: dict[str, str] = {v.name: v.description for v in config.variants}
    columns: list[str] = _table_metrics(table)
    view: pd.DataFrame = table[columns].copy()
    view.columns = [info(m).title for m in columns]
    view.insert(0, "What it is", [descriptions.get(n, "") for n in table.index])
    view.index = [f"{_marker(n, rec)}{n}" for n in table.index]
    view.index.name = "Configuration"

    def shade(col: pd.Series) -> list[str]:
        metric: str | None = next((m for m in columns if info(m).title == col.name), None)
        if metric is None:
            return ["" for _ in col]
        values: pd.Series = col if info(metric).higher_is_better else -col
        lo, hi = float(values.min()), float(values.max())
        styles: list[str] = []
        for v in values:
            step: int = 0 if hi == lo else round((float(v) - lo) / (hi - lo) * (len(SEQUENTIAL) - 1))
            styles.append(f"background-color: {SEQUENTIAL[step]}; color: {'white' if step >= 3 else TEXT_PRIMARY}")
        return styles

    formats: dict[str, Any] = {info(m).title: (lambda v, m=m: fmt(m, v)) for m in columns}
    return view.style.apply(shade).format(formats).set_table_styles(
        [{"selector": "th", "props": [("text-align", "left"), ("padding", "6px 10px")]},
         {"selector": "td", "props": [("padding", "6px 10px")]}]
    )


def key_findings(table: pd.DataFrame, rec: Recommendation, reranked: set[str] | None = None) -> list[str]:
    """Plain-English takeaways, generated from the numbers. `reranked` names the variants that already rerank."""
    findings: list[str] = []

    def best(metric: str) -> str | None:
        if metric not in table:
            return None
        series: pd.Series = table[metric] if info(metric).higher_is_better else -table[metric]
        return str(series.idxmax())

    if (b := best("recall_at_10")) is not None:
        findings.append(
            f"<b>Best at finding the correct document</b>: {b} has it in the top 10 results for "
            f"{fmt('recall_at_10', _value(table, b, 'recall_at_10'))} of questions."
        )
    if (b := best("recall_at_1")) is not None:
        findings.append(
            f"<b>Best at ranking it first</b>: {b} returns the correct document as the #1 result for "
            f"{fmt('recall_at_1', _value(table, b, 'recall_at_1'))} of questions."
        )
    speed: str = time_metric(table)
    if (b := best(speed)) is not None:
        slowest: str = str(table[speed].idxmax())
        findings.append(
            f"<b>Fastest</b>: {b}, {fmt(speed, _value(table, b, speed))} per search on average. "
            f"Slowest: {slowest}, {fmt(speed, _value(table, slowest, speed))}."
        )
    if {"recall_at_10", "recall_at_1"} <= set(table.columns):
        found, first = _value(table, rec.baseline, "recall_at_10"), _value(table, rec.baseline, "recall_at_1")
        findings.append(
            f"<b>Today's setup</b> ({rec.baseline}) finds the correct document in the top 10 for {fmt('recall_at_10', found)} "
            f"of questions and ranks it first for {fmt('recall_at_1', first)}."
        )
        found, first = _value(table, rec.variant, "recall_at_10"), _value(table, rec.variant, "recall_at_1")
        if found - first >= 0.2:
            fix: str = (
                "a different or stronger reranker, or more candidates for it to choose from"
                if rec.variant in (reranked or set())
                else "a reranker or a different search mode"
            )
            findings.append(
                f"<b>Finding vs. ranking</b>: with {rec.variant}, the correct document is in the top 10 for "
                f"{fmt('recall_at_10', found)} of questions but first for only {fmt('recall_at_1', first)}. If users or the "
                f"application rely on the first result, {fix} may close that gap."
            )
    if "retrieval_relevance" in table and "recall_at_10" in table:
        findings.append(
            f"<b>AI judge</b>: on average {fmt('retrieval_relevance', float(table['retrieval_relevance'].mean()))} of returned "
            "results are judged relevant. This is lower than the found-in-top-10 rate because each search returns several "
            "results, and only some can be about the specific item asked for."
        )
    if rec.num_questions < 100:
        findings.append(
            f"<b>Test size</b>: these results come from {rec.num_questions} test questions. Differences of a few points "
            "can be chance at this size (see the margin of error in the ranking chart)."
        )
    return findings


def reranked_variants(config: ExperimentConfig) -> set[str]:
    """Variants whose retriever already reranks (built-in reranker columns, or a reranking factory)."""
    return {
        v.name for v in config.variants
        if v.retriever.params.get("rerank_columns") or "rerank" in v.retriever.factory.rsplit(":", 1)[-1]
    }


def key_findings_html(table: pd.DataFrame, rec: Recommendation, config: ExperimentConfig | None = None) -> str:
    reranked: set[str] = reranked_variants(config) if config else set()
    items: str = "".join(f"<li style='margin-bottom: 6px'>{f}</li>" for f in key_findings(table, rec, reranked))
    return f"<div style='font-family: sans-serif; color: {TEXT_PRIMARY}; max-width: 960px'><ul>{items}</ul></div>"


def _param_value(key: str, value: str) -> str:
    if key == "retriever.query_type":
        return SEARCH_MODES.get(value, value)
    if key == "index_source":
        return "Existing index (already in use)" if value == "existing" else "New index built for this test"
    return value


def recommendation_html(rec: Recommendation, table: pd.DataFrame, variant: Variant, params: dict[str, str]) -> str:
    """The recommendation in plain language. `params` are the MLflow params logged on the recommended run."""
    p: MetricInfo = info(rec.primary_metric)
    s: MetricInfo = info(rec.secondary_metric)
    esc = html.escape

    # Why
    why: list[str] = []
    if rec.variant == rec.baseline:
        why.append(f"No configuration did clearly better than today's setup on <b>{p.label.lower()}</b> ({p.technical}).")
    else:
        why.append(
            f"<b>{p.label}</b> ({p.technical}), the measure this recommendation optimizes: "
            f"{fmt(rec.primary_metric, _value(table, rec.variant, rec.primary_metric))}, compared with "
            f"{fmt(rec.primary_metric, _value(table, rec.baseline, rec.primary_metric))} for today's setup ({esc(rec.baseline)}), "
            f"a change of {fmt_change(rec.primary_metric, rec.change, long=True)}."
        )
    if rec.variant != rec.leader:
        same: bool = abs(_value(table, rec.leader, rec.primary_metric) - _value(table, rec.variant, rec.primary_metric)) < 1e-9
        why.append(
            f"{esc(rec.leader)} scored {'the same' if same else 'slightly higher, but within the margin of error,'} on "
            f"{p.label.lower()}, so the tie was broken by <b>{s.label.lower()}</b> ({s.technical}), where {esc(rec.variant)} "
            "is better."
        )
    speed: str = time_metric(table)
    for metric in ("recall_at_1", speed):
        if metric in table and metric != rec.primary_metric and rec.variant != rec.baseline:
            m = info(metric)
            why.append(
                f"<b>{m.label}</b> ({m.technical}): {fmt(metric, _value(table, rec.variant, metric))}, compared with "
                f"{fmt(metric, _value(table, rec.baseline, metric))} today."
            )
    slowdown: float = _value(table, rec.variant, speed) / max(_value(table, rec.baseline, speed), 1e-9)
    if rec.variant != rec.baseline and slowdown >= SLOWDOWN_WARNING:
        why.append(
            f"<b>Trade-off: about {slowdown:.0f}× slower than today</b> ({fmt(speed, _value(table, rec.variant, speed))} vs "
            f"{fmt(speed, _value(table, rec.baseline, speed))} per search). If that is too slow for your application, set "
            f"<code>latency_budget_ms</code> (for example {_value(table, rec.baseline, speed) * SLOWDOWN_WARNING:.0f}) and "
            "re-run this notebook to get the best option within it."
        )
    if rec.excluded_by_latency:
        why.append(
            f"Not considered because they are slower than the {rec.latency_budget_ms:.0f} ms limit: "
            f"{esc(', '.join(rec.excluded_by_latency))}."
        )

    # Confidence
    confidence: list[str] = []
    margin: str = fmt_change(rec.primary_metric, rec.margin, long=True).lstrip("+")
    if rec.variant == rec.baseline:
        confidence.append("Keep the current configuration; none of the alternatives is clearly better on this test.")
    elif rec.clear_improvement and rec.change >= CLEAR_MARGIN * rec.margin:
        confidence.append(
            f"<b>High.</b> The improvement ({fmt_change(rec.primary_metric, rec.change, long=True)}) is well beyond the margin "
            f"of error (±{margin}), so it is very unlikely to be chance."
        )
    elif rec.clear_improvement:
        confidence.append(
            f"<b>Moderate.</b> The improvement ({fmt_change(rec.primary_metric, rec.change, long=True)}) is only just larger "
            f"than the margin of error (±{margin}). It is probably real, but a larger test would make it certain."
        )
    else:
        confidence.append(
            f"<b>Low to moderate.</b> The improvement ({fmt_change(rec.primary_metric, rec.change, long=True)}) is smaller than "
            f"the margin of error (±{margin}) for {rec.num_questions} test questions, so it could be chance."
        )
    others: list[str] = [t for t in rec.tied if t != rec.variant]
    if others:
        names: str = ", ".join(f"{t} (today's setup)" if t == rec.baseline else t for t in others)
        confidence.append(
            f"On {p.label.lower()}, this test can't yet separate the recommended configuration from: {esc(names)}. "
            "They scored within the margin of error of each other."
        )

    # Next steps
    steps: list[str] = []
    if not rec.clear_improvement or rec.change < CLEAR_MARGIN * rec.margin or rec.num_questions < 100:
        steps.append(
            "Confirm with a larger test: set <code>num_evals</code> to 200 or more and <code>overwrite_dataset</code> to "
            "<code>true</code>, then re-run all three notebooks."
        )
    steps.append("Open the recommended run in MLflow (Evaluations tab) and read a few questions it missed.")
    if rec.variant != rec.baseline:
        steps.append("When satisfied, apply the configuration below in your application.")

    rows: str = "".join(
        f"<tr><td style='padding: 4px 16px 4px 0'>{info(m).label} <span style='color: {TEXT_SECONDARY}'>({info(m).technical})</span></td>"
        f"<td style='padding: 4px 16px 4px 0'><b>{fmt(m, _value(table, rec.variant, m))}</b></td>"
        f"<td style='padding: 4px 16px 4px 0'>{fmt(m, _value(table, rec.baseline, m))}</td>"
        f"<td style='padding: 4px 0'>{fmt_change(m, _value(table, rec.variant, m) - _value(table, rec.baseline, m))}</td></tr>"
        for m in _table_metrics(table)
    )
    shown: dict[str, str] = {k: v for k, v in params.items() if k != "k" and v not in (None, "", "None")}
    if "chunk_size" not in shown:
        shown.pop("chunk_overlap", None)  # overlap is meaningless without chunking
    ordered: list[str] = [k for k in PARAM_ORDER if k in shown] + sorted(k for k in shown if k not in PARAM_ORDER)
    config: str = "".join(
        f"<tr><td style='padding: 3px 16px 3px 0'>{PARAM_LABELS.get(k, k)}</td><td><code>{esc(_param_value(k, shown[k]))}</code></td></tr>"
        for k in ordered
    )

    def section(title: str, items: list[str]) -> str:
        return f"<h3 style='margin: 16px 0 4px'>{title}</h3><ul style='margin-top: 4px'>" + "".join(
            f"<li style='margin-bottom: 4px'>{i}</li>" for i in items) + "</ul>"

    return f"""
<div style="font-family: sans-serif; color: {TEXT_PRIMARY}; max-width: 960px; line-height: 1.45">
  <h2 style="margin-bottom: 2px">Recommended configuration: <span style="color: {RECOMMENDED}">{esc(rec.variant)}</span></h2>
  <p style="color: {TEXT_SECONDARY}; margin-top: 0">{esc(variant.description)}</p>
  {section("Why", why)}
  {section("How confident is this?", confidence)}
  {section("What to do next", steps)}
  <h3 style="margin: 16px 0 4px">Compared with today's setup</h3>
  <table style="border-collapse: collapse">
    <tr style="text-align: left; color: {TEXT_SECONDARY}"><th style="padding: 4px 16px 4px 0">Measure</th>
      <th style="padding: 4px 16px 4px 0">{esc(rec.variant)} (recommended)</th>
      <th style="padding: 4px 16px 4px 0">{esc(rec.baseline)} (today)</th><th>Change</th></tr>
    {rows}
  </table>
  <h3 style="margin: 16px 0 4px">Configuration</h3>
  <table style="border-collapse: collapse">{config}</table>
  <p style="color: {TEXT_SECONDARY}; font-size: 0.9em; margin-top: 16px">How this was chosen: the best <b>{p.label.lower()}</b>
  ({p.technical}){f", among configurations within the {rec.latency_budget_ms:.0f} ms limit" if rec.latency_budget_ms else ""}.
  Configurations too close to call (within the margin of error) are separated by <b>{s.label.lower()}</b> ({s.technical}),
  then by speed. To optimize for something else, change <code>primary_metric</code>, <code>secondary_metric</code>,
  <code>latency_budget_ms</code> or <code>baseline_variant</code> and re-run this notebook.</p>
</div>"""


# ---------------------------------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------------------------------


def _role_color(name: str, rec: Recommendation) -> str:
    return RECOMMENDED if name == rec.variant else BASELINE if name == rec.baseline else OTHER


def _role(name: str, rec: Recommendation) -> str:
    return " (recommended)" if name == rec.variant else " (today)" if name == rec.baseline else ""


def _axis_format(metric: str) -> dict[str, Any]:
    kind: Kind = info(metric).kind
    return {"tickformat": ".0%"} if kind == "pct" else {"ticksuffix": " ms"} if kind == "ms" else {"tickformat": ".2f"}


def _layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"color": TEXT_PRIMARY, "size": 16}},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font={"color": TEXT_SECONDARY, "size": 12},
        margin={"l": 60, "r": 30, "t": 60, "b": 50}, hoverlabel={"bgcolor": "white", "font_color": TEXT_PRIMARY},
        legend={"orientation": "h", "y": -0.2},
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _interval(table: pd.DataFrame, metric: str) -> tuple[list[float], list[float]]:
    """95% interval half-widths (plus, minus), clamped to [0, 1] for metrics that are proportions or scores."""
    mean: pd.Series = table[metric]
    half: pd.Series = Z_95 * table[f"{metric}_se"].fillna(0)
    if info(metric).kind == "ms":
        return list(half), list(half.clip(upper=mean))
    return list(half.clip(upper=1 - mean)), list(half.clip(upper=mean))


def ranking_chart(table: pd.DataFrame, rec: Recommendation) -> go.Figure:
    """The primary metric per configuration, with its margin of error. Values sit in the axis labels so they never
    collide with the whiskers."""
    metric: str = rec.primary_metric
    m: MetricInfo = info(metric)
    order: pd.DataFrame = table.sort_values(metric, ascending=m.higher_is_better)
    plus, minus = _interval(order, metric)
    labels: list[str] = [f"{n}{_role(n, rec)}  {fmt(metric, order.at[n, metric])}" for n in order.index]
    fig: go.Figure = go.Figure(go.Bar(
        x=order[metric], y=labels, orientation="h",
        marker={"color": [_role_color(n, rec) for n in order.index], "line": {"color": SURFACE, "width": 2}},
        error_x={"type": "data", "symmetric": False, "array": plus, "arrayminus": minus, "color": TEXT_SECONDARY,
                 "thickness": 1.5, "width": 6},
        customdata=[[n, fmt(metric, v), fmt(metric, v - lo), fmt(metric, v + hi)]
                    for n, v, lo, hi in zip(order.index, order[metric], minus, plus)],
        hovertemplate="%{customdata[0]}<br>" + m.label + ": %{customdata[1]}<br>"
        "Likely range: %{customdata[2]} – %{customdata[3]}<extra></extra>",
    ))
    fig.update_layout(height=max(300, 45 * len(order) + 140))
    if m.kind != "ms":
        fig.update_xaxes(range=[0, 1.0])
    fig.update_xaxes(**_axis_format(metric))
    return _layout(fig, f"{m.title} by configuration", f"{m.label}: whiskers show the likely range (95% margin of error)", "")


def recall_curve_chart(table: pd.DataFrame, rec: Recommendation) -> go.Figure:
    """Share of questions with the correct document in the top 1, 5 and 10 results."""
    fig: go.Figure = go.Figure()
    ks: list[int] = [k for k in CUTOFFS if f"recall_at_{k}" in table]
    others: list[str] = [n for n in table.index if n not in (rec.variant, rec.baseline)]
    for i, name in enumerate(others + [rec.baseline, rec.variant]):  # highlighted lines drawn last, on top
        role: str = "Recommended" if name == rec.variant else "Today's setup" if name == rec.baseline else "Other configurations"
        fig.add_trace(go.Scatter(
            x=[f"Top {k}" for k in ks], y=[table.at[name, f"recall_at_{k}"] for k in ks],
            name=f"{role}: {name}" if role != "Other configurations" else role,
            mode="lines+markers", line={"color": _role_color(name, rec), "width": 2 if role != "Other configurations" else 1.5},
            marker={"size": 8}, legendgroup=role, showlegend=role != "Other configurations" or i == 0,
            hovertemplate=f"{name}<br>Correct document in the %{{x}}: %{{y:.0%}} of questions<extra></extra>",
        ))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    return _layout(fig, "How often is the correct document in the top 1, 5 and 10 results?", "", "Share of questions")


def quality_latency_chart(table: pd.DataFrame, rec: Recommendation) -> go.Figure:
    """Primary metric vs search time. Only the recommended and baseline points are labeled directly (configurations
    often overlap); hover names the rest."""
    metric: str = rec.primary_metric
    m: MetricInfo = info(metric)
    speed: str = time_metric(table)
    plus, minus = _interval(table, metric)
    fig: go.Figure = go.Figure()
    groups: list[tuple[str, list[str]]] = [
        ("Other configurations", [n for n in table.index if n not in (rec.variant, rec.baseline)]),
        (f"Today's setup: {rec.baseline}", [rec.baseline] if rec.baseline != rec.variant else []),
        (f"Recommended: {rec.variant}", [rec.variant]),
    ]
    for label, names in groups:
        if not names:
            continue
        idx: list[int] = [list(table.index).index(n) for n in names]
        highlighted: bool = not label.startswith("Other")
        fig.add_trace(go.Scatter(
            x=table.loc[names, speed], y=table.loc[names, metric], name=label,
            mode="markers+text" if highlighted else "markers", text=names if highlighted else None,
            textposition="top center", textfont={"color": TEXT_PRIMARY, "size": 12},
            marker={"size": 12, "color": _role_color(names[0], rec), "line": {"color": SURFACE, "width": 2}},
            error_y={"type": "data", "symmetric": False, "array": [plus[i] for i in idx],
                     "arrayminus": [minus[i] for i in idx], "color": GRID, "thickness": 1.5, "width": 4},
            customdata=[[n, fmt(metric, table.at[n, metric]), fmt(speed, table.at[n, speed])] for n in names],
            hovertemplate="%{customdata[0]}<br>" + m.label + ": %{customdata[1]}<br>Search time: %{customdata[2]}<extra></extra>",
        ))
    lat: pd.Series = table[speed]
    pad: float = max(float(lat.max() - lat.min()) * 0.12, 25.0)
    fig.update_xaxes(range=[max(0.0, float(lat.min()) - pad), float(lat.max()) + pad], ticksuffix=" ms")
    fig.update_yaxes(**_axis_format(metric))
    return _layout(fig, f"{m.label} vs. search time (top left is best: better and faster)",
                   f"Average {info(speed).label.lower()} per question", m.title)


def chart_html(fig: go.Figure) -> str:
    """A chart as self-contained HTML for `displayHTML` (plotly.js from the Plotly CDN).

    More reliable than `fig.show()` in Databricks notebooks and jobs, where the interactive renderer's output is
    occasionally not captured.
    """
    return fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displaylogo": False, "responsive": True})


def analyze(config: ExperimentConfig, summary: pd.DataFrame, num_questions: int) -> tuple[pd.DataFrame, Recommendation]:
    table: pd.DataFrame = metrics_table(summary, num_questions)
    return table, recommend(table, config.settings, config.selected_variants(), num_questions)


def tag_runs(summary: pd.DataFrame, rec: Recommendation) -> None:
    """Mark the recommended and baseline runs, so they can be filtered in the experiment UI."""
    client: mlflow.MlflowClient = mlflow.MlflowClient()
    for name, run_id in summary["run_id"].items():
        role: str = "recommended" if name == rec.variant else "baseline" if name == rec.baseline else "candidate"
        client.set_tag(run_id, "retrieval_eval.role", role)
