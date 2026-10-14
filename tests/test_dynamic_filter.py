from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from retrieval_eval.config import ExperimentConfig
from retrieval_eval.factories import DynamicFilterRetriever, filters_to_dict, guard_filters

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
