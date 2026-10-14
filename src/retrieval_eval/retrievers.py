"""Running the retriever under test.

Each variant names a factory, `factory(settings, variant, **params)`, that returns a LangChain `BaseRetriever` or
`BaseTool` (see `retrieval_eval.factories` for examples). `RetrievalTarget` wraps whatever the factory returns in one
MLflow RETRIEVER span with normalized documents (`metadata.doc_uri` set), which the scorers and judges read.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import mlflow
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from mlflow.entities import Document, SpanType

from retrieval_eval.config import ExperimentConfig, Variant

RetrieverComponent = BaseRetriever | BaseTool
RetrieverFactory = Callable[..., RetrieverComponent]


def normalize_doc_uri(value: Any) -> str:
    """Match Spark's string cast of the id column (numeric ids can come back as floats, e.g. 123.0)."""
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


# ---------------------------------------------------------------------------------------------------------------------
# Factory loading and result normalization
# ---------------------------------------------------------------------------------------------------------------------


def load_factory(spec: str, base_dir: Path) -> RetrieverFactory:
    """Resolve 'package.module:function' or 'path/to/file.py:function' (relative to `base_dir`)."""
    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        raise ValueError(f"retriever factory must look like 'module:function' or 'file.py:function', got {spec!r}")
    module: ModuleType
    if target.endswith(".py"):
        path: Path = (base_dir / target).resolve()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load retriever factory file {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    factory: RetrieverFactory = getattr(module, attr)
    return factory


def _documents(value: Any) -> list[tuple[str, dict[str, Any]]] | None:
    """(page_content, metadata) pairs from LangChain Documents, dicts, or a JSON string of either; None if not documents."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    pairs: list[tuple[str, dict[str, Any]]] = []
    for item in value:
        if isinstance(item, LCDocument):
            pairs.append((item.page_content, dict(item.metadata)))
        elif isinstance(item, dict):
            metadata: dict[str, Any] = item.get("metadata") or {k: v for k, v in item.items() if k != "page_content"}
            pairs.append((str(item.get("page_content") or item.get("content") or ""), dict(metadata)))
        else:
            return None
    return pairs


class RetrievalTarget:
    """The `predict_fn` for one variant: build its retriever once, then answer each query inside a RETRIEVER span."""

    def __init__(self, config: ExperimentConfig, variant: Variant) -> None:
        self.variant: Variant = variant
        factory: RetrieverFactory = load_factory(variant.retriever.factory, config.base_dir)
        self.component: RetrieverComponent = factory(config.settings, variant, **variant.retriever.params)
        if not isinstance(self.component, (BaseRetriever, BaseTool)):
            raise TypeError(
                f"{variant.retriever.factory} returned {type(self.component).__name__}; expected a LangChain BaseRetriever "
                "or BaseTool"
            )

    def _invoke(self, query: str) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(self.component, BaseRetriever):
            return [(d.page_content, dict(d.metadata)) for d in self.component.invoke(query)]
        # Invoke as a tool call so tools with response_format="content_and_artifact" (e.g. LangChain's
        # create_retriever_tool) return their Documents as the message artifact.
        message: Any = self.component.invoke(
            {"name": self.component.name, "args": {self.variant.retriever.query_arg: query}, "id": "retrieval-eval", "type": "tool_call"}
        )
        candidates: list[Any] = [message.artifact, message.content] if isinstance(message, ToolMessage) else [message]
        for candidate in candidates:
            documents = _documents(candidate)
            if documents is not None:
                return documents
        raise TypeError(
            f"tool {self.component.name!r} did not return documents (got {type(candidates[-1]).__name__}). Return a "
            "BaseRetriever from the factory, or a tool that returns Documents, dicts, JSON, or a Documents artifact."
        )

    def retrieve(self, query: str) -> int:
        """One untraced search (for warm-up and timing); returns the number of results."""
        return len(self._invoke(query))

    def warm_up(self, query: str) -> None:
        """One untraced call, so connection setup and scale-from-zero are not billed to any measured search."""
        self.retrieve(query)

    @mlflow.trace(name="retrieve", span_type=SpanType.RETRIEVER)
    def search(self, query: str) -> list[Document]:
        key: str = self.variant.retriever.doc_uri_key
        documents: list[Document] = []
        for page_content, metadata in self._invoke(query):
            if metadata.get(key) is None:
                raise KeyError(f"retrieved document has no metadata[{key!r}]; set retriever.doc_uri_key (keys: {sorted(metadata)})")
            documents.append(Document(page_content=page_content, metadata={**metadata, "doc_uri": normalize_doc_uri(metadata[key])}))
        return documents
