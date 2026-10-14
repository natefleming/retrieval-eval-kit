"""Read source documents behind an existing AI Search index."""

from __future__ import annotations

from databricks.sdk import WorkspaceClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from retrieval_eval.config import Settings


def source_table(settings: Settings, w: WorkspaceClient | None = None) -> str:
    """The Delta table a Delta Sync index is built from."""
    index = (w or WorkspaceClient()).vector_search_indexes.get_index(settings.source_index)
    if not index.delta_sync_index_spec or not index.delta_sync_index_spec.source_table:
        raise ValueError(f"{settings.source_index} is not a Delta Sync index; point source_index at one that is")
    return index.delta_sync_index_spec.source_table


def source_endpoint(settings: Settings, w: WorkspaceClient | None = None) -> str:
    """`settings.vs_endpoint`, or the endpoint that serves the source index."""
    if settings.vs_endpoint:
        return settings.vs_endpoint
    endpoint: str | None = (w or WorkspaceClient()).vector_search_indexes.get_index(settings.source_index).endpoint_name
    if not endpoint:
        raise ValueError(f"cannot determine the endpoint of {settings.source_index}; set vs_endpoint")
    return endpoint


def describe_index(index_name: str, w: WorkspaceClient | None = None) -> dict[str, str]:
    """What an index embeds and with which model, read from its definition (for logging and reports)."""
    spec = (w or WorkspaceClient()).vector_search_indexes.get_index(index_name).delta_sync_index_spec
    if spec is None:
        return {"embedding_model": "self-managed (Direct Vector Access)"}
    if spec.embedding_source_columns:
        source = spec.embedding_source_columns[0]
        return {"embedding_model": source.embedding_model_endpoint_name or "unknown", "embedded_text": source.name or "unknown"}
    if spec.embedding_vector_columns:
        return {"embedding_model": "self-managed vectors", "embedded_text": spec.embedding_vector_columns[0].name or "unknown"}
    return {}


def documents(spark: SparkSession, settings: Settings, columns: list[str], separator: str = "\n") -> DataFrame:
    """One row per source document: `doc_uri` (string) and `content` (the columns joined by `separator`)."""
    return spark.table(source_table(settings)).select(
        F.col(settings.doc_uri_column).cast("string").alias("doc_uri"),
        F.concat_ws(separator, *[F.col(c).cast("string") for c in columns]).alias("content"),
    ).where(F.length("content") > 0)
