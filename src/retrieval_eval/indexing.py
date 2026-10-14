"""Build one AI Search index per distinct (text recipe, embedding model) in the experiment.

Each new index is a Delta Sync index with managed embeddings over a chunk table:
`<output_schema>.rek_<hash>_chunks` (chunk_id, doc_uri, content) → `<output_schema>.rek_<hash>_index`.
Building is idempotent: an existing chunk table or index is reused, so re-running only builds what is missing.
"""

from __future__ import annotations

from databricks.ai_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from retrieval_eval.config import ExperimentConfig, Settings, TextConfig, Variant
from retrieval_eval.sources import documents, source_endpoint


def chunk(docs: DataFrame, text: TextConfig) -> DataFrame:
    """Split `content` into chunks (or keep documents whole) with stable ids `<doc_uri>_<n>`."""
    if text.chunk_size:
        size: int = text.chunk_size
        overlap: int = text.chunk_overlap

        @F.udf(ArrayType(StringType()))
        def split(content: str) -> list[str]:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(content)

        docs = docs.select("doc_uri", F.posexplode(split("content")).alias("n", "content"))
    else:
        docs = docs.withColumn("n", F.lit(0))
    return docs.select(F.concat_ws("_", "doc_uri", F.col("n").cast("string")).alias("chunk_id"), "doc_uri", "content")


def index_exists(w: WorkspaceClient, name: str) -> bool:
    try:
        w.vector_search_indexes.get_index(name)
        return True
    except NotFound:
        return False


def build_index(spark: SparkSession, settings: Settings, variant: Variant, endpoint: str) -> str:
    index_name: str | None = variant.index_name(settings)
    index = variant.index
    if index_name is None or index is None or index.text is None or index.embedding_model is None:
        raise ValueError(f"variant {variant.name!r} does not describe an index to build")
    w: WorkspaceClient = WorkspaceClient()
    vsc: VectorSearchClient = VectorSearchClient(disable_notice=True)
    table: str = index_name.removesuffix("_index") + "_chunks"
    text: TextConfig = index.text

    if not spark.catalog.tableExists(table):
        print(f"Writing chunk table {table}")
        chunk(documents(spark, settings, text.columns, text.separator), text).write.saveAsTable(table)
        spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")  # required by Delta Sync
    if not index_exists(w, index_name):
        print(f"Creating index {index_name} on {endpoint} with {index.embedding_model}")
        vsc.create_delta_sync_index(
            endpoint_name=endpoint,
            index_name=index_name,
            primary_key="chunk_id",
            source_table_name=table,
            pipeline_type="TRIGGERED",
            embedding_source_column="content",
            embedding_model_endpoint_name=index.embedding_model,
        )
    vsc.get_index(index_name=index_name).wait_until_ready(verbose=True)
    return index_name


def build_all(spark: SparkSession, config: ExperimentConfig) -> list[str]:
    """Build every index the selected variants need (each distinct recipe once). Returns the index names."""
    settings: Settings = config.settings
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {settings.output_schema}")
    endpoint: str = source_endpoint(settings)
    to_build: dict[str, Variant] = {v.index.key: v for v in config.selected_variants() if v.index and not v.index.existing}
    return [build_index(spark, settings, v, endpoint) for v in to_build.values()]
