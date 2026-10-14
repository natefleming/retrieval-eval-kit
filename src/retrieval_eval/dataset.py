"""Generate a synthetic MLflow evaluation dataset from the documents behind an existing AI Search index."""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.genai.datasets
import pandas as pd
from databricks.agents.evals import generate_evals_df
from mlflow.genai.datasets import EvaluationDataset
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from retrieval_eval.config import Settings
from retrieval_eval.sources import documents


def to_record(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one `generate_evals_df` row into the record shape used by this kit.

    `inputs` becomes `{"query": ...}` (the retriever's only argument). Expectations keep the source doc's `doc_uri`
    (ground truth for the IR metrics) and the generated `expected_facts` (used by RetrievalSufficiency).
    `source_type`/`source_id` are passed through so the dataset's `source` column records the source document.
    """
    messages: list[dict[str, Any]] = row["inputs"]["messages"]
    expectations: dict[str, Any] = row["expectations"]
    return {
        "inputs": {"query": messages[-1]["content"]},
        "expectations": {
            "expected_retrieved_context": [
                {"doc_uri": c["doc_uri"], "content": c.get("content")} for c in expectations["expected_retrieved_context"]
            ],
            "expected_facts": list(expectations.get("expected_facts") or []),
        },
        "source_type": row["source_type"],
        "source_id": row["source_id"],
    }


def existing_dataset(name: str) -> EvaluationDataset | None:
    try:
        return mlflow.genai.datasets.get_dataset(name=name)
    except Exception:  # noqa: BLE001 - dataset does not exist
        return None


def generate(spark: SparkSession, settings: Settings) -> EvaluationDataset:
    """Create the UC-backed dataset from `num_evals` questions generated over `num_docs` sampled documents.

    An existing dataset is kept as-is, so repeated runs evaluate against the same questions and stay comparable.
    Set `overwrite_dataset` to delete it and generate a new one.
    """
    dataset = existing_dataset(settings.dataset_name)
    if dataset is not None and not settings.overwrite_dataset:
        print(f"Using existing dataset {settings.dataset_name} (set overwrite_dataset=true to regenerate)")
        return dataset
    if dataset is not None:
        mlflow.genai.datasets.delete_dataset(name=settings.dataset_name)
    docs: pd.DataFrame = (
        documents(spark, settings, settings.content_columns).orderBy(F.rand(seed=42)).limit(settings.num_docs).toPandas()
    )
    evals: pd.DataFrame = generate_evals_df(
        docs,
        num_evals=settings.num_evals,
        agent_description=settings.agent_description,
        question_guidelines=settings.question_guidelines or None,
    )
    records: list[dict[str, Any]] = [to_record(r) for r in evals.to_dict("records")]

    return mlflow.genai.datasets.create_dataset(name=settings.dataset_name).merge_records(records)
