from collections.abc import Callable
from pathlib import Path
import mlflow
import pytest
from mlflow.entities import Document, SpanType, Trace


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A throwaway local tracking store, so tests produce real traces without a workspace. Rebuilding a custom scorer
    from source (as Databricks does for registered scorers) needs an explicit opt-in outside Databricks."""
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("tests") is None:
        mlflow.create_experiment("tests", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("tests")


def documents(doc_uris: list[str]) -> list[Document]:
    return [Document(page_content=f"doc {u}", metadata={"doc_uri": u}) for u in doc_uris]


@pytest.fixture
def make_trace() -> Callable[..., Trace]:
    """Trace a retrieval returning `doc_uris`; `nested` adds an inner RETRIEVER span (as wrapped retrievers emit)."""

    def build(doc_uris: list[str], nested: list[str] | None = None) -> Trace:
        @mlflow.trace(name="inner", span_type=SpanType.RETRIEVER)
        def inner() -> list[Document]:
            return documents(nested or [])

        @mlflow.trace(name="retrieve", span_type=SpanType.RETRIEVER)
        def retrieve(query: str) -> list[Document]:
            if nested is not None:
                inner()
            return documents(doc_uris)

        retrieve("q")
        mlflow.flush_trace_async_logging()
        trace: Trace | None = mlflow.get_trace(mlflow.get_last_active_trace_id())
        assert trace is not None
        return trace

    return build


