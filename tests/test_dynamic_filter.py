from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from retrieval_eval.config import ExperimentConfig
import mlflow

from retrieval_eval.factories import (
    DynamicFilterRetriever,
    FusionSearchRetriever,
    filters_to_dict,
    guard_filters,
    reciprocal_rank_fusion,
)

VOCAB: dict[str, dict[str, list[str]]] = {
    "brand_name": {"dewalt": ["DEWALT"], "milwaukee": ["Milwaukee"], "big green egg": ["BIG GREEN EGG", "Big Green Egg"]},
    "merchandise_class": {"plant food": ["PLANT FOOD"]},
}


class FakePlanner:
    """Stands in for an LLM bound to the tool: always 'calls' it with the given arguments."""

    def __init__(self, args: dict[str, Any]) -> None:
        self.args = args

    def invoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": "search", "args": self.args, "id": "1"}])


class FakeStore:
    """Records searches; returns nothing when a filter is applied if `empty_when_filtered`."""

    def __init__(self, empty_when_filtered: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.empty_when_filtered = empty_when_filtered

    def similarity_search(self, query: str, k: int, filter: dict[str, Any] | None, query_type: str) -> list[Document]:
        self.calls.append((query, filter))
        if filter and self.empty_when_filtered:
            return []
        return [Document(page_content="p", metadata={"product_id": 1})]


def retriever(args: dict[str, Any], store: FakeStore, guarded: bool) -> DynamicFilterRetriever:
    return DynamicFilterRetriever.model_construct(
        planner=FakePlanner(args), store=store, k=10, query_type="HYBRID", vocabulary=VOCAB if guarded else None
    )


def test_filters_to_dict_matches_the_tool_format() -> None:
    assert filters_to_dict([{"key": "brand_name NOT", "value": "DEWALT"}, {"key": "x", "value": [1, 2]}]) == {
        "brand_name NOT": "DEWALT", "x": [1, 2]}
    assert filters_to_dict(None) == {}


def test_guard_fixes_casing_drops_unknown_values_and_passes_other_columns() -> None:
    planned = {"brand_name NOT": "dewalt", "merchandise_class": "Plant Food", "brand_name": ["Milwaukee", "Acme"],
               "price <": 50}
    assert guard_filters(planned, VOCAB) == {
        "brand_name NOT": "DEWALT", "merchandise_class": "PLANT FOOD", "brand_name": ["Milwaukee"], "price <": 50}
    assert guard_filters({"merchandise_class": "Gardening"}, VOCAB) == {}


def test_guard_expands_to_every_stored_spelling() -> None:
    """The catalog stores some brands in several casings; a filter must match all of them, include or exclude."""
    assert guard_filters({"brand_name": "Big Green Egg"}, VOCAB) == {"brand_name": ["BIG GREEN EGG", "Big Green Egg"]}
    assert guard_filters({"brand_name NOT": "big green egg"}, VOCAB) == {"brand_name NOT": ["BIG GREEN EGG", "Big Green Egg"]}


def test_unguarded_uses_the_llm_filters_exactly() -> None:
    store = FakeStore()
    retriever({"query": "impact driver", "filters": [{"key": "brand_name NOT", "value": "dewalt"}]}, store, False).invoke(
        "impact driver, not DeWalt")
    assert store.calls == [("impact driver", {"brand_name NOT": "dewalt"})]


def test_guarded_fixes_values_and_falls_back_to_unfiltered_when_empty() -> None:
    store = FakeStore(empty_when_filtered=True)
    docs = retriever({"query": "fertilizer", "filters": [{"key": "merchandise_class", "value": "plant food"}]}, store,
                     True).invoke("fertilizer from plant food")
    assert store.calls == [("fertilizer", {"merchandise_class": "PLANT FOOD"}), ("fertilizer", None)]
    assert len(docs) == 1


def test_filters_config_uses_its_own_dataset_and_resolves_factories() -> None:
    from retrieval_eval.retrievers import load_factory

    config = ExperimentConfig.load(Path(__file__).parents[1] / "config" / "filters.yaml")
    main = ExperimentConfig.load(Path(__file__).parents[1] / "config" / "experiment.yaml")
    assert config.settings.dataset_name != main.settings.dataset_name
    assert config.settings.overwrite_dataset is False and "merchandise_class" in config.settings.content_columns
    for variant in config.variants:
        load_factory(variant.retriever.factory, config.base_dir)
    guarded = [v for v in config.variants if v.retriever.params.get("guarded")]
    assert guarded and all(v.retriever.params.get("guard_columns") for v in guarded)


def test_no_tool_call_falls_back_to_the_plain_request() -> None:
    class Silent:
        def invoke(self, messages: list[Any]) -> AIMessage:
            return AIMessage(content="Here are some drills...")

    store = FakeStore()
    r = DynamicFilterRetriever.model_construct(planner=Silent(), store=store, k=10, query_type="HYBRID", vocabulary=None)
    assert r.plan("cordless drill") == {"query": "cordless drill", "filters": {}, "used_tool": False}
    r.invoke("cordless drill")
    assert store.calls == [("cordless drill", None)]


def doc(product_id: int) -> Document:
    return Document(page_content=f"p{product_id}", metadata={"product_id": product_id})


def test_reciprocal_rank_fusion_rewards_documents_found_by_both_searches() -> None:
    fused = reciprocal_rank_fusion([[doc(1), doc(2), doc(3)], [doc(3), doc(4)]], "product_id", k=3)
    assert [d.metadata["product_id"] for d in fused] == [3, 1, 2]  # 3 is in both lists; 1 leads one; 2 ties 4, seen first
    assert fused[0].metadata["fusion_score"] == 1 / 63 + 1 / 61


class TypedStore(FakeStore):
    """Returns different results per query type, and records which types were searched."""

    RESULTS: dict[str, list[int]] = {"FULL_TEXT": [1, 2], "HYBRID": [2, 3]}

    def __init__(self, empty_when_filtered: bool = False) -> None:
        super().__init__(empty_when_filtered)
        self.query_types: list[str] = []

    def similarity_search(self, query: str, k: int, filter: dict[str, Any] | None, query_type: str) -> list[Document]:
        self.calls.append((query, filter))
        self.query_types.append(query_type)
        if filter and self.empty_when_filtered:
            return []
        return [doc(i) for i in self.RESULTS[query_type]]


def fusion(store: TypedStore, args: dict[str, Any]) -> FusionSearchRetriever:
    return FusionSearchRetriever.model_construct(
        planner=FakePlanner(args), store=store, k=10, query_type="HYBRID", vocabulary=VOCAB,
        query_types=["FULL_TEXT", "HYBRID"], id_key="product_id",
    )


def test_fusion_runs_each_query_type_with_the_guarded_filters_and_fuses() -> None:
    store = TypedStore()
    docs = fusion(store, {"query": "impact driver", "filters": [{"key": "brand_name NOT", "value": "dewalt"}]}).invoke(
        "impact driver, not DeWalt")
    assert sorted(store.query_types) == ["FULL_TEXT", "HYBRID"]
    assert store.calls == [("impact driver", {"brand_name NOT": "DEWALT"})] * 2
    assert [d.metadata["product_id"] for d in docs] == [2, 1, 3]


def test_fusion_falls_back_to_unfiltered_when_every_filtered_search_is_empty() -> None:
    store = TypedStore(empty_when_filtered=True)
    docs = fusion(store, {"query": "fertilizer", "filters": [{"key": "merchandise_class", "value": "plant food"}]}).invoke(
        "fertilizer from plant food")
    assert [f for _, f in store.calls] == [{"merchandise_class": "PLANT FOOD"}] * 2 + [None] * 2
    assert len(docs) == 3


def test_fusion_searches_are_spans_in_the_callers_trace() -> None:
    @mlflow.trace(name="retrieve")
    def run() -> None:
        fusion(TypedStore(), {"query": "drill"}).invoke("drill")

    run()
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    names = {span.name for span in trace.data.spans}
    assert {"retrieve", "plan_filters", "search_full_text", "search_hybrid"} <= names
    assert len(mlflow.search_traces(return_type="list")) == 1  # the threads' spans did not start traces of their own
