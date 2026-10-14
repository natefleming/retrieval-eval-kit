"""Evaluate each variant against the MLflow evaluation dataset: one MLflow run per variant."""

from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.genai.datasets
import pandas as pd
from mlflow.entities import Experiment
from mlflow.genai.datasets import EvaluationDataset
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.scorers import Scorer

from retrieval_eval.config import ExperimentConfig, Settings, Variant
from retrieval_eval.retrievers import RetrievalTarget
from retrieval_eval.sources import describe_index
from retrieval_eval.scorers import make_scorers, register_scorers


def per_question_variance(result_df: pd.DataFrame) -> dict[str, float]:
    """`<metric>/variance` from the per-question `<metric>/value` columns (judge yes/no as 1/0).

    MLflow aggregates only the mean for scorers that return several feedbacks, and the analysis needs the spread to put
    a margin of error on each metric.
    """
    variances: dict[str, float] = {}
    for column in result_df.columns:
        if column.endswith("/value"):
            values: pd.Series = pd.to_numeric(
                result_df[column].map(lambda v: 1.0 if v in (True, "yes") else 0.0 if v in (False, "no") else v),
                errors="coerce",
            ).dropna()
            if len(values) > 1:
                variances[column.removesuffix("/value") + "/variance"] = float(values.var(ddof=1))
    return variances


def measure_search_time(target: RetrievalTarget, queries: list[str]) -> dict[str, float]:
    """Time searches one at a time, so the result is the time a single user waits.

    `mlflow.genai.evaluate` sends many questions in parallel, which makes trace latency (`latency_ms`) reflect queueing
    under load (dramatic for a single-replica GPU reranker) rather than the time of one search.
    """
    times: list[float] = []
    for query in queries:
        start: float = time.perf_counter()
        target.retrieve(query)
        times.append((time.perf_counter() - start) * 1000)
    if not times:
        return {}
    ordered: list[float] = sorted(times)
    return {
        "search_time_ms/mean": statistics.fmean(times),
        "search_time_ms/p90": ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))],
        "search_time_ms/variance": statistics.variance(times) if len(times) > 1 else 0.0,
    }


def evaluate_variant(config: ExperimentConfig, variant: Variant, scorers: list[Scorer], sweep_id: str) -> dict[str, Any]:
    settings: Settings = config.settings
    target: RetrievalTarget = RetrievalTarget(config, variant)
    dataset: EvaluationDataset = mlflow.genai.datasets.get_dataset(name=settings.dataset_name)
    queries: list[str] = [r["query"] for r in dataset.to_df()["inputs"]]
    target.warm_up(queries[0] if queries else "warm up")  # cold start (e.g. scale-from-zero) is not billed to any search
    with mlflow.start_run(run_name=variant.name, description=variant.description or None):
        mlflow.set_tags({"retrieval_eval.sweep_id": sweep_id, "retrieval_eval.dataset": settings.dataset_name})
        index_name: str | None = variant.index_name(settings)
        mlflow.log_params(variant.params(settings, describe_index(index_name) if index_name else None))
        mlflow.log_dict(variant.model_dump(), "variant.json")
        result: EvaluationResult = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=target.search,
            scorers=scorers,
        )
        search_time: dict[str, float] = measure_search_time(target, queries[: settings.search_time_samples])
        mlflow.log_metrics({k: v for k, v in search_time.items() if not k.endswith("/variance")})
    return {
        "variant": variant.name, "run_id": result.run_id, **result.metrics, **per_question_variance(result.result_df),
        **search_time,
    }


def run(config: ExperimentConfig, experiment_name: str) -> pd.DataFrame:
    """Evaluate the selected variants in the active experiment; one row of aggregate metrics per variant, best recall first.

    The scorers are registered to the experiment first, so they are visible (and reusable) on its Scorers tab.
    """
    scorers: list[Scorer] = make_scorers(config.settings.judge_model, config.settings.use_llm_judges)
    experiment: Experiment | None = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"experiment {experiment_name} does not exist; call setup_mlflow first")
    register_scorers(scorers, experiment.experiment_id)
    sweep_id: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = [evaluate_variant(config, v, scorers, sweep_id) for v in config.selected_variants()]
    summary: pd.DataFrame = pd.DataFrame(rows).set_index("variant")
    return summary.sort_values("recall_at_10/mean", ascending=False) if "recall_at_10/mean" in summary else summary
