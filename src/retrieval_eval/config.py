"""Experiment configuration: shared settings plus the retrieval variants to compare.

Everything is loaded from one YAML file (`config/experiment.yaml`). Job parameters and notebook widgets can override any
`settings` value; an empty override means "use the value from the file".
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

class Settings(BaseModel):
    """Settings shared by every step and every variant."""

    source_index: str = Field(description="Existing AI Search index (catalog.schema.index) used as the document source")
    doc_uri_column: str = Field(description="Column that identifies a source document; it becomes the eval `doc_uri`")
    content_columns: list[str] = Field(description="Columns concatenated into document text for question generation")
    output_schema: str = Field(description="catalog.schema where the dataset, chunk tables and new indexes are written")
    vs_endpoint: str | None = Field(None, description="AI Search endpoint for new indexes; defaults to the source index's")
    dataset_table: str = Field("retrieval_eval_dataset", description="MLflow evaluation dataset table name in output_schema")
    num_docs: int = Field(200, description="Documents sampled from the source to generate questions from")
    num_evals: int = Field(100, description="Synthetic questions to generate (MLflow datasets hold at most 2,000 rows)")
    agent_description: str = Field(description="What the retrieval system is for; guides synthetic question generation")
    question_guidelines: str = Field("", description="Style and audience guidelines for synthetic questions")
    overwrite_dataset: bool = Field(False, description="Delete and regenerate the dataset; otherwise an existing one is reused")
    k: int = Field(10, description="Default number of results per query, for factories that take no explicit count")
    search_time_samples: int = Field(
        20, description="Questions timed one at a time (after warm-up) for the reported search time; 0 to skip"
    )
    judge_model: str = Field("databricks:/databricks-claude-sonnet-5-5", description="LLM judge as provider:/endpoint")
    use_llm_judges: bool = Field(True, description="Add MLflow RetrievalRelevance and RetrievalSufficiency judges")
    primary_metric: str = Field("recall_at_10", description="Metric the recommendation optimizes (e.g. recall_at_1, ndcg_at_10)")
    secondary_metric: str = Field("ndcg_at_10", description="Tie-breaker between variants within the margin of error")
    latency_budget_ms: float | None = Field(None, description="Exclude variants slower than this mean latency from the recommendation")
    baseline_variant: str | None = Field(None, description="Variant the recommendation is compared against; default the first")
    experiment_name: str | None = Field(None, description="MLflow experiment path; defaults to /Users/<me>/retrieval-eval-kit-experiment")
    variants: list[str] = Field(default_factory=list, description="Variant names to run; empty runs all")

    @field_validator("content_columns", "variants", mode="before")
    @classmethod
    def _split_csv(cls, value: Any) -> Any:
        """Accept "a,b" or "a b" (spaces are handy because `bundle run --params` splits on commas)."""
        return [v for v in re.split(r"[,\s]+", value) if v] if isinstance(value, str) else value

    @property
    def dataset_name(self) -> str:
        return f"{self.output_schema}.{self.dataset_table}"


class TextConfig(BaseModel):
    """How document text is built and chunked for a new index."""

    columns: list[str] = Field(description="Source columns concatenated (in order) into the text that is embedded")
    separator: str = Field("\n", description="Separator placed between concatenated columns")
    chunk_size: int | None = Field(None, description="Characters per chunk; omit to embed each document whole")
    chunk_overlap: int = Field(0, description="Characters shared by consecutive chunks")


class IndexConfig(BaseModel):
    """Either an existing index, or the recipe for a new one."""

    existing: str | None = Field(None, description="Existing index name; nothing is built")
    text: TextConfig | None = Field(None, description="Text/chunking recipe for a new index")
    embedding_model: str | None = Field(None, description="Embedding endpoint for a new index (managed embeddings)")

    @model_validator(mode="after")
    def _one_kind(self) -> IndexConfig:
        if bool(self.existing) == bool(self.text and self.embedding_model):
            raise ValueError("index needs either `existing`, or both `text` and `embedding_model`")
        return self

    @property
    def key(self) -> str:
        """Stable hash of a build recipe, so variants that differ only at query time share one index."""
        recipe = {"text": self.text.model_dump() if self.text else None, "embedding_model": self.embedding_model}
        return hashlib.sha1(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:10]


class RetrieverConfig(BaseModel):
    """Which LangChain retriever or tool answers the queries, and how to read document ids from its results."""

    factory: str = Field(
        description="'package.module:function' or 'path/to/file.py:function' (relative to the experiment file). Called as "
        "factory(settings, variant, **params) and must return a LangChain BaseRetriever or BaseTool",
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Keyword arguments passed to the factory (e.g. query_type)")
    doc_uri_key: str = Field("doc_uri", description="Document metadata key holding the id matched against ground truth")
    query_arg: str = Field("query", description="Argument name the query is passed as when the factory returns a tool")


class Variant(BaseModel):
    """One retrieval configuration; each variant becomes one MLflow run."""

    name: str = Field(description="Unique name, used as the MLflow run name")
    description: str = Field("", description="Free-text note shown on the MLflow run")
    retriever: RetrieverConfig = Field(description="Factory that builds the LangChain retriever or tool under test")
    index: IndexConfig | None = Field(None, description="Index this variant queries; new recipes are built by step 2")

    def index_name(self, settings: Settings) -> str | None:
        if self.index is None:
            return None
        return self.index.existing or f"{settings.output_schema}.rek_{self.index.key}_index"

    def params(self, settings: Settings, index_info: dict[str, str] | None = None) -> dict[str, Any]:
        """Flat params logged to MLflow, so runs can be compared and filtered in the UI (factory params as `retriever.*`).

        `index_info` (from `sources.describe_index`) supplies what the config can't know about an existing index, such as
        its embedding model and embedded column. Params without a value are left out rather than logged as None.
        """
        info: dict[str, str] = index_info or {}
        text: TextConfig | None = self.index.text if self.index else None
        chunking: str | None = None
        if text is not None:
            chunking = f"{text.chunk_size} chars, {text.chunk_overlap} overlap" if text.chunk_size else "whole document"
        elif self.index is not None:
            chunking = "as built (existing index)"
        params: dict[str, Any] = {
            "retriever": self.retriever.factory,
            **{f"retriever.{k}": v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str)
               for k, v in self.retriever.params.items() if v is not None},
            "index": self.index_name(settings),
            "index_source": None if self.index is None else "existing" if self.index.existing else "built",
            # A built index's recipe is more descriptive than its definition (which only names the chunk-table column).
            "embedding_model": (self.index.embedding_model if self.index else None) or info.get("embedding_model"),
            "embedded_text": (" + ".join(text.columns) if text else None) or info.get("embedded_text"),
            "chunking": chunking,
            "chunk_size": text.chunk_size if text else None,
            "chunk_overlap": text.chunk_overlap if text and text.chunk_size else None,
            "k": settings.k,
        }
        return {k: v for k, v in params.items() if v is not None}


class ExperimentConfig(BaseModel):
    settings: Settings
    variants: list[Variant]
    base_dir: Path = Field(Path("."), exclude=True, description="Directory of the experiment file; file factories resolve from here")

    @field_validator("variants")
    @classmethod
    def _unique_names(cls, variants: list[Variant]) -> list[Variant]:
        names = [v.name for v in variants]
        if len(names) != len(set(names)):
            raise ValueError(f"variant names must be unique: {names}")
        return variants

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
        """Load the YAML file, apply non-empty overrides to `settings`, and resolve `${setting}` references in variants.

        A file may set `base: other.yaml` (relative to itself) to inherit that file's settings, overriding only the ones
        it lists, and its variants when it defines none. This keeps focused comparisons (e.g. config/rerankers.yaml) on
        the same dataset and settings as the main experiment.
        """
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        if raw.get("base"):
            base: dict[str, Any] = yaml.safe_load((Path(path).parent / raw.pop("base")).read_text())
            raw = {"settings": {**base.get("settings", {}), **raw.get("settings", {})},
                   "variants": raw.get("variants") or base.get("variants", [])}
        raw["settings"] = {**raw.get("settings", {}), **{k: v for k, v in (overrides or {}).items() if v not in (None, "")}}
        text = json.dumps(raw["variants"])
        for key, value in raw["settings"].items():
            if isinstance(value, str):
                text = text.replace("${" + key + "}", value)
        raw["variants"] = json.loads(text)
        return cls.model_validate({**raw, "base_dir": Path(path).resolve().parent})

    def selected_variants(self) -> list[Variant]:
        if not self.settings.variants:
            return self.variants
        unknown = set(self.settings.variants) - {v.name for v in self.variants}
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}")
        return [v for v in self.variants if v.name in self.settings.variants]
