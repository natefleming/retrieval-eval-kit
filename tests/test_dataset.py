"""Dataset generation runs only when the dataset is missing, or when overwrite_dataset is true."""

import sys
import types
from typing import Any

import pytest

from retrieval_eval.config import Settings


@pytest.fixture
def dataset_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import retrieval_eval.dataset with pyspark stubbed (pyspark is provided by the Databricks runtime)."""
    for name in ("pyspark", "pyspark.sql", "pyspark.sql.functions"):
        stub = types.ModuleType(name)
        stub.SparkSession = stub.DataFrame = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.delitem(sys.modules, "retrieval_eval.dataset", raising=False)
    monkeypatch.delitem(sys.modules, "retrieval_eval.sources", raising=False)
    import retrieval_eval.dataset as module

    return module


def settings(**overrides: Any) -> Settings:
    return Settings.model_validate({"source_index": "c.s.i", "doc_uri_column": "id", "content_columns": ["t"],
                                    "output_schema": "c.o", "agent_description": "test", **overrides})


def test_existing_dataset_is_reused_and_nothing_is_generated(dataset_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = object()
    calls: list[str] = []
    monkeypatch.setattr(dataset_module, "existing_dataset", lambda name: existing)
    monkeypatch.setattr(dataset_module, "documents", lambda *a, **k: calls.append("documents"))
    monkeypatch.setattr(dataset_module, "generate_evals_df", lambda *a, **k: calls.append("generate_evals_df"))
    monkeypatch.setattr(dataset_module.mlflow.genai.datasets, "delete_dataset", lambda **k: calls.append("delete"))

    assert dataset_module.generate(spark=None, settings=settings()) is existing
    assert calls == []  # no sampling, no generation, no deletion


def test_overwrite_deletes_and_regenerates(dataset_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(dataset_module, "existing_dataset", lambda name: object())
    monkeypatch.setattr(dataset_module.mlflow.genai.datasets, "delete_dataset", lambda **k: calls.append("delete"))

    def stop(*args: Any, **kwargs: Any) -> None:
        calls.append("documents")
        raise RuntimeError("stop after deciding to regenerate")

    monkeypatch.setattr(dataset_module, "documents", stop)
    with pytest.raises(RuntimeError, match="stop after"):
        dataset_module.generate(spark=None, settings=settings(overwrite_dataset=True))
    assert calls == ["delete", "documents"]
