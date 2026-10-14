"""Example retriever factories. Each returns a LangChain `BaseRetriever` or `BaseTool` for one variant.

A variant references a factory as `retrieval_eval.factories:<name>` (or `path/to/your_file.py:<name>` for your own,
relative to the experiment file). It is called as `factory(settings, variant, **retriever.params)`:

    retriever:
      factory: retrieval_eval.factories:ai_search
      params: {query_type: HYBRID, rerank_columns: [product_name, description]}

Included: `ai_search` (AI Search client), `databricks_vector_search` and `retriever_tool` (the LangChain components
existing apps use), `endpoint_reranker` / `endpoint_reranker_tool` (AI Search candidates re-scored by a custom
reranker on Model Serving, as a retriever or as an agent tool), and `dynamic_filter` (an LLM writes metadata filters
per request, as an agent using `VectorSearchRetrieverTool(dynamic_filter=True)` would). Reranking always lives inside the retriever, so callers, and the
evaluator, see one `query → documents` step.
To evaluate your own retriever, copy one of these into a file in the repo, change it, and point a variant at it.
"""

from __future__ import annotations

from typing import Any, Literal

import mlflow

from databricks.ai_search.client import VectorSearchClient
from databricks.ai_search.index import VectorSearchIndex
from databricks.ai_search.reranker import DatabricksReranker
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch, VectorSearchRetrieverTool
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from databricks.sdk import WorkspaceClient
from langchain_core.tools.retriever import create_retriever_tool
from mlflow.entities import SpanType
from pydantic import ConfigDict, PrivateAttr

from retrieval_eval.config import Settings, Variant
from retrieval_eval.retrievers import normalize_doc_uri

QueryType = Literal["ANN", "HYBRID", "FULL_TEXT"]


# ---------------------------------------------------------------------------------------------------------------------
# ai_search: the AI Search Python client directly (full control over query type, reranker, filters, threshold)
# ---------------------------------------------------------------------------------------------------------------------


class AISearchRetriever(BaseRetriever):
    """Queries the variant's AI Search index (`index.existing`, or the index built from `index.text`).

    Built indexes expose `doc_uri`/`content`; existing indexes expose `settings.doc_uri_column` plus
    `settings.content_columns`, which are joined into `page_content`. Results carry `metadata.doc_uri`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index_name: str
    doc_uri_column: str
    content_columns: list[str]
    query_type: QueryType = "ANN"
    num_results: int = 10
    rerank_columns: list[str] = []
    filters: dict[str, Any] | str | None = None
    score_threshold: float | None = None
    _index: VectorSearchIndex = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._index = VectorSearchClient(disable_notice=True).get_index(index_name=self.index_name)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        kwargs: dict[str, Any] = {
            "query_text": query,
            "columns": [self.doc_uri_column, *self.content_columns],
            "num_results": self.num_results,
            "query_type": self.query_type,
            "filters": self.filters,
            "score_threshold": self.score_threshold,
        }
        if self.rerank_columns:
            kwargs["reranker"] = DatabricksReranker(columns_to_rerank=self.rerank_columns)
        response: dict[str, Any] = self._index.similarity_search(**kwargs)

        names: list[str] = [c["name"] for c in response.get("manifest", {}).get("columns", [])]
        rows: list[dict[str, Any]] = [dict(zip(names, r)) for r in response.get("result", {}).get("data_array") or []]
        return [
            Document(
                page_content="\n".join(str(row[c]) for c in self.content_columns if row.get(c) is not None),
                metadata={"doc_uri": normalize_doc_uri(row[self.doc_uri_column]), "score": row.get("score")},
            )
            for row in rows
        ]


def ai_search(
    settings: Settings,
    variant: Variant,
    *,
    query_type: QueryType = "ANN",
    num_results: int | None = None,
    rerank_columns: list[str] | None = None,
    filters: dict[str, Any] | str | None = None,
    score_threshold: float | None = None,
) -> BaseRetriever:
    """AI Search over the variant's index.

    Args:
        query_type: ANN (vector), HYBRID (vector + keyword) or FULL_TEXT (keyword).
        num_results: Results per query; defaults to `settings.k`.
        rerank_columns: Enable the Databricks reranker on these columns.
        filters: Dict for STANDARD endpoints, SQL string for STORAGE_OPTIMIZED endpoints.
        score_threshold: Drop results scoring below this value.
    """
    index_name: str | None = variant.index_name(settings)
    if index_name is None:
        raise ValueError(f"variant {variant.name!r}: ai_search needs an `index`")
    built: bool = variant.index is not None and not variant.index.existing
    return AISearchRetriever(
        index_name=index_name,
        doc_uri_column="doc_uri" if built else settings.doc_uri_column,
        content_columns=["content"] if built else settings.content_columns,
        query_type=query_type,
        num_results=num_results or settings.k,
        rerank_columns=rerank_columns or [],
        filters=filters,
        score_threshold=score_threshold,
    )


# ---------------------------------------------------------------------------------------------------------------------
# databricks_vector_search / retriever_tool: the LangChain components existing Databricks RAG apps and agents use
# ---------------------------------------------------------------------------------------------------------------------


def databricks_vector_search(
    settings: Settings,
    variant: Variant,
    *,
    query_type: QueryType = "ANN",
    num_results: int | None = None,
    columns: list[str] | None = None,
) -> BaseRetriever:
    """A `databricks_langchain.DatabricksVectorSearch` retriever over the variant's index (default: `source_index`).

    Returned documents carry the index columns in `metadata`, so set the variant's `doc_uri_key` to the id column.
    """
    store: DatabricksVectorSearch = DatabricksVectorSearch(
        index_name=variant.index_name(settings) or settings.source_index,
        columns=columns or [settings.doc_uri_column, *settings.content_columns],
    )
    return store.as_retriever(search_kwargs={"k": num_results or settings.k, "query_type": query_type})


def as_retriever_tool(retriever: BaseRetriever, name: str, description: str) -> BaseTool:
    """Wrap any retriever (including a reranking one) as the LangChain retriever *tool* an agent calls.

    Whatever the retriever does internally (vector search, hybrid, reranking) stays behind the tool's single `query`
    argument. `response_format="content_and_artifact"` keeps the Documents (with metadata) as the tool's artifact,
    which the evaluator reads; a tool that returns only text loses the document ids.
    """
    return create_retriever_tool(retriever, name=name, description=description, response_format="content_and_artifact")


def retriever_tool(
    settings: Settings,
    variant: Variant,
    *,
    name: str = "search",
    description: str = "Search the knowledge base for documents relevant to the request.",
    **params: Any,
) -> BaseTool:
    """`databricks_vector_search` as a retriever tool. `params` go to `databricks_vector_search`."""
    return as_retriever_tool(databricks_vector_search(settings, variant, **params), name, description)


# ---------------------------------------------------------------------------------------------------------------------
# endpoint_reranker: AI Search candidates re-scored by a custom reranker on Model Serving (Qwen3, gte, bge, ...)
# ---------------------------------------------------------------------------------------------------------------------


class EndpointRerankRetriever(BaseRetriever):
    """Two-stage retrieval: `base` returns a candidate pool, a reranker endpoint scores each (query, document) pair, and
    the `top_n` highest-scoring candidates are returned in score order (`metadata.rerank_score` holds the score).

    Endpoint contract (an MLflow pyfunc cross-encoder, as deployed by notebooks/04_deploy_rerankers.py):
    `{"dataframe_records": [{"query": str, "document": str}, ...]}` → `{"predictions": [float, ...]}`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base: BaseRetriever
    endpoint_name: str
    top_n: int = 10
    document_chars: int = 600
    _client: WorkspaceClient | None = PrivateAttr(default=None)

    @mlflow.trace(name="rerank", span_type=SpanType.RERANKER)
    def score(self, query: str, documents: list[str]) -> list[float]:
        if self._client is None:  # created on first use, so building the retriever needs no credentials
            self._client = WorkspaceClient()
        records: list[dict[str, str]] = [{"query": query, "document": d[: self.document_chars]} for d in documents]
        response: Any = self._client.api_client.do(
            "POST", f"/serving-endpoints/{self.endpoint_name}/invocations", body={"dataframe_records": records}
        )
        return [float(s) for s in response["predictions"]]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        candidates: list[Document] = self.base.invoke(query)
        if not candidates:
            return []
        scores: list[float] = self.score(query, [d.page_content for d in candidates])
        ranked: list[tuple[float, Document]] = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        return [
            Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": s}) for s, d in ranked[: self.top_n]
        ]


def endpoint_reranker(
    settings: Settings,
    variant: Variant,
    *,
    endpoint_name: str,
    candidates: int = 50,
    num_results: int | None = None,
    document_chars: int = 600,
    **search_params: Any,
) -> BaseRetriever:
    """AI Search fetches `candidates` results; a custom reranker on Model Serving re-scores them, and the best
    `num_results` (default `settings.k`) are returned. Deploy rerankers with notebooks/04_deploy_rerankers.py.

    Args:
        endpoint_name: Model Serving endpoint with the cross-encoder contract above (e.g. retrieval-qwen3-reranker-0-6b).
            Any model works as long as it accepts (query, document) records and returns one score per record.
        candidates: How many AI Search results to rerank. More candidates find more, but each costs reranker time.
        document_chars: Characters of each result sent to the reranker (cross-encoders have a token limit).
        search_params: Passed to `ai_search` for the candidate search, e.g. `query_type: HYBRID`.
    """
    return EndpointRerankRetriever(
        base=ai_search(settings, variant, num_results=candidates, **search_params),
        endpoint_name=endpoint_name,
        top_n=num_results or settings.k,
        document_chars=document_chars,
    )


def endpoint_reranker_tool(
    settings: Settings,
    variant: Variant,
    *,
    name: str = "search",
    description: str = "Search the knowledge base for documents relevant to the request.",
    **params: Any,
) -> BaseTool:
    """`endpoint_reranker` as a retriever tool: the agent calls one `search` tool, and candidate search plus reranking
    happen inside it. `params` go to `endpoint_reranker` (endpoint_name, candidates, query_type, ...)."""
    return as_retriever_tool(endpoint_reranker(settings, variant, **params), name, description)


# ---------------------------------------------------------------------------------------------------------------------
# dynamic_filter: an LLM writes metadata filters per request, as an agent would with VectorSearchRetrieverTool
# ---------------------------------------------------------------------------------------------------------------------

PLANNER_PROMPT: str = (
    "You search a catalog for the user. Call the search tool once. Put the core need in `query`, and add `filters` "
    "only for explicit constraints in the request (e.g. a brand to include or exclude, a category). Do not invent "
    "constraints the user did not state."
)


def filters_to_dict(filters: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The tool's `filters` argument ([{key, value}, ...]) as the dict AI Search expects, e.g. {"brand_name NOT": "X"}."""
    return {str(f["key"]): f["value"] for f in filters or [] if isinstance(f, dict) and "key" in f}


Vocabulary = dict[str, dict[str, list[str]]]  # column → {lowercased value: every spelling stored in the column}


def guard_filters(filters: dict[str, Any], vocabulary: Vocabulary) -> dict[str, Any]:
    """Keep filters only when their values exist in the column, expanded to every stored spelling; filters on other
    columns pass through.

    Catalogs often store one value in several casings (e.g. "Big Green Egg" and "BIG GREEN EGG"), so an exact-match
    filter on one spelling silently misses the rest. A filter key is "column" or "column OPERATOR" (e.g. "brand NOT").
    """
    guarded: dict[str, Any] = {}
    for key, value in filters.items():
        column: str = key.split()[0]
        if column not in vocabulary:
            guarded[key] = value
            continue
        values: list[Any] = value if isinstance(value, list) else [value]
        spellings: list[str] = sorted({s for v in values for s in vocabulary[column].get(str(v).lower(), [])})
        if spellings:
            guarded[key] = spellings if isinstance(value, list) or len(spellings) > 1 else spellings[0]
    return guarded


class DynamicFilterRetriever(BaseRetriever):
    """Plans `query` + `filters` with an LLM bound to the real dynamic-filter tool schema, then searches with them.

    The search runs through `DatabricksVectorSearch` (the store the tool wraps) so results keep their metadata; the tool
    itself returns plain text. The planning step is traced as its own span, so each question's filters are visible.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    planner: Any  # an LLM with the tool bound (a LangChain Runnable)
    store: DatabricksVectorSearch
    k: int = 10
    query_type: QueryType = "HYBRID"
    vocabulary: Vocabulary | None = None  # set → guarded mode

    @mlflow.trace(name="plan_filters", span_type=SpanType.CHAIN)
    def plan(self, request: str) -> dict[str, Any]:
        message: AIMessage = self.planner.invoke([SystemMessage(PLANNER_PROMPT), HumanMessage(request)])
        # tool_choice is left to the model (several, e.g. Claude Sonnet 5.5, reject forcing a tool). If it answers
        # without calling the tool, search with the original request and no filters.
        args: dict[str, Any] = message.tool_calls[0]["args"] if message.tool_calls else {}
        filters: dict[str, Any] = filters_to_dict(args.get("filters"))
        planned: dict[str, Any] = {"query": args.get("query") or request, "filters": filters, "used_tool": bool(message.tool_calls)}
        if self.vocabulary is not None:
            planned["guarded_filters"] = guard_filters(filters, self.vocabulary)
        return planned

    def _search(self, query: str, filters: dict[str, Any]) -> list[Document]:
        return self.store.similarity_search(query, k=self.k, filter=filters or None, query_type=self.query_type)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        planned: dict[str, Any] = self.plan(query)
        filters: dict[str, Any] = planned.get("guarded_filters", planned["filters"])
        documents: list[Document] = self._search(planned["query"], filters)
        if not documents and filters and self.vocabulary is not None:
            documents = self._search(planned["query"], {})  # guarded: never return nothing because of a filter
        return documents


def _column_vocabulary(table: str, columns: list[str], max_values: int = 20000) -> Vocabulary:
    """Distinct values per column from the source table (needs an active Spark session, as in notebooks and jobs)."""
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("guarded dynamic filters need a Spark session to read column values")
    vocabulary: Vocabulary = {}
    for column in columns:
        rows = spark.table(table).select(column).where(f"{column} IS NOT NULL").distinct().limit(max_values).collect()
        spellings: dict[str, list[str]] = {}
        for row in rows:
            spellings.setdefault(str(row[0]).lower(), []).append(str(row[0]))
        vocabulary[column] = spellings
    return vocabulary


def dynamic_filter(
    settings: Settings,
    variant: Variant,
    *,
    llm_endpoint: str = "databricks-claude-sonnet-5-5",
    llm_temperature: float | None = None,
    query_type: QueryType = "HYBRID",
    num_results: int | None = None,
    columns: list[str] | None = None,
    guarded: bool = False,
    guard_columns: list[str] | None = None,
    rerank_endpoint: str | None = None,
    candidates: int = 50,
) -> BaseRetriever:
    """Dynamic metadata filtering, evaluated the way an agent uses it.

    `VectorSearchRetrieverTool(dynamic_filter=True)` only *exposes* a `filters` argument; the calling agent's LLM fills
    it in. This factory plays that agent: `llm_endpoint` sees the real tool (schema and column descriptions) and writes
    `query` + `filters` for each request.

    Args:
        llm_endpoint: Chat model that plans the filters (an agent's model).
        llm_temperature: Sampling temperature, if the model accepts one (e.g. Claude Sonnet 5.5 does not); omitted by default.
        columns: Columns returned and described to the LLM; default `doc_uri_column` + `content_columns`.
        guarded: Check filter values against the column's actual values (fix casing, drop unknown values) and fall
            back to unfiltered search when a filtered search returns nothing.
        guard_columns: Categorical columns whose filter values are checked in guarded mode (required when guarded),
            e.g. [brand_name, merchandise_class]. Filters on other columns pass through unchanged.
        rerank_endpoint: Optional custom reranker endpoint (see `endpoint_reranker`) applied to `candidates` results.
    """
    from retrieval_eval.sources import source_table

    index_name: str = variant.index_name(settings) or settings.source_index
    cols: list[str] = columns or [settings.doc_uri_column, *settings.content_columns]
    tool: VectorSearchRetrieverTool = VectorSearchRetrieverTool(
        index_name=index_name, columns=cols, num_results=num_results or settings.k, query_type=query_type,
        dynamic_filter=True, tool_name="search", tool_description="Search the catalog for items matching the request.",
    )
    llm: BaseChatModel = (
        ChatDatabricks(endpoint=llm_endpoint)
        if llm_temperature is None
        else ChatDatabricks(endpoint=llm_endpoint, temperature=llm_temperature)
    )
    vocabulary: Vocabulary | None = None
    if guarded:
        if not guard_columns:
            raise ValueError("dynamic_filter: guarded=true needs guard_columns, e.g. [brand_name, merchandise_class]")
        vocabulary = _column_vocabulary(source_table(settings), guard_columns)
    retriever: BaseRetriever = DynamicFilterRetriever(
        planner=llm.bind_tools([tool]),
        store=DatabricksVectorSearch(index_name=index_name, columns=cols),
        k=candidates if rerank_endpoint else (num_results or settings.k), query_type=query_type, vocabulary=vocabulary,
    )
    if rerank_endpoint:
        return EndpointRerankRetriever(base=retriever, endpoint_name=rerank_endpoint, top_n=num_results or settings.k)
    return retriever
