from pathlib import Path
from typing import Any

import pytest

from retrieval_eval.config import ExperimentConfig, Settings, Variant
from retrieval_eval.retrievers import RetrievalTarget, load_factory

FIXTURES = Path(__file__).parent / "fixtures"
SETTINGS = Settings(
    source_index="c.s.idx", doc_uri_column="id", content_columns=["text"], output_schema="c.out", agent_description="test"
)


def target(factory: str, params: dict[str, Any], **retriever: Any) -> RetrievalTarget:
    variant = Variant(name="v", retriever={"factory": f"factories.py:{factory}", "params": params, **retriever})
    return RetrievalTarget(ExperimentConfig(settings=SETTINGS, variants=[variant], base_dir=FIXTURES), variant)


def uris(t: RetrievalTarget) -> list[str]:
    return [d.metadata["doc_uri"] for d in t.search("q")]


def test_load_factory_by_file_and_module(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_factory("factories.py:fixed", FIXTURES).__name__ == "fixed"
    monkeypatch.syspath_prepend(str(FIXTURES))
    assert load_factory("factories:fixed", Path("/nowhere")).__name__ == "fixed"
    with pytest.raises(ValueError, match="module:function"):
        load_factory("no_colon", FIXTURES)


def test_base_retriever() -> None:
    assert uris(target("fixed", {"ids": ["a", "b"]})) == ["a", "b"]


def test_doc_uri_key_maps_metadata_and_normalizes_numbers() -> None:
    t = target("fixed", {"ids": [12.0, "x"], "key": "product_id"}, doc_uri_key="product_id")
    assert uris(t) == ["12", "x"]


def test_missing_doc_uri_key_is_a_clear_error() -> None:
    with pytest.raises(KeyError, match="doc_uri_key"):
        target("fixed", {"ids": ["a"], "key": "sku"}).search("q")


def test_tool_with_documents_artifact() -> None:
    assert uris(target("artifact_tool", {"ids": ["a", "b"]})) == ["a", "b"]


def test_tool_with_json_content() -> None:
    assert uris(target("json_tool", {"ids": ["a"]})) == ["a"]


def test_tool_without_documents_is_a_clear_error() -> None:
    with pytest.raises(TypeError, match="did not return documents"):
        target("text_tool", {}).search("q")


def test_factory_must_return_langchain_component() -> None:
    with pytest.raises(TypeError, match="expected a LangChain"):
        target("not_a_retriever", {})


def test_every_factory_in_the_example_config_resolves() -> None:
    config = ExperimentConfig.load(Path(__file__).parents[1] / "config" / "experiment.yaml")
    for variant in config.variants:
        assert callable(load_factory(variant.retriever.factory, config.base_dir)), variant.name


def test_ai_search_needs_an_index() -> None:
    from retrieval_eval.factories import ai_search

    with pytest.raises(ValueError, match="needs an `index`"):
        ai_search(SETTINGS, Variant(name="v", retriever={"factory": "retrieval_eval.factories:ai_search"}))


def test_measure_search_time_runs_one_at_a_time_untraced() -> None:
    import mlflow

    from retrieval_eval.evaluate import measure_search_time

    t = target("fixed", {"ids": ["a"]})
    before = mlflow.get_last_active_trace_id()
    times = measure_search_time(t, ["q1", "q2", "q3"])
    assert set(times) == {"search_time_ms/mean", "search_time_ms/p90", "search_time_ms/variance"}
    assert mlflow.get_last_active_trace_id() == before and times["search_time_ms/mean"] >= 0


def test_warm_up_is_not_traced() -> None:
    import mlflow

    t = target("fixed", {"ids": ["a"]})
    before = mlflow.get_last_active_trace_id()
    t.warm_up("q")
    assert mlflow.get_last_active_trace_id() == before


def test_endpoint_reranker_reorders_candidates_by_score(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.documents import Document as LCDocument

    from retrieval_eval.factories import EndpointRerankRetriever
    from tests.fixtures.factories import FixedRetriever

    sent: list[list[str]] = []

    def fake_score(self: EndpointRerankRetriever, query: str, documents: list[str]) -> list[float]:
        sent.append(documents)
        return [0.1, 0.9, 0.5]  # candidates a, b, c → b first, then c

    monkeypatch.setattr(EndpointRerankRetriever, "score", fake_score)
    retriever = EndpointRerankRetriever(base=FixedRetriever(ids=["a", "b", "c"]), endpoint_name="qwen", top_n=2)
    docs: list[LCDocument] = retriever.invoke("q")
    assert [d.metadata["doc_uri"] for d in docs] == ["b", "c"]
    assert docs[0].metadata["rerank_score"] == 0.9 and sent == [["q a", "q b", "q c"]]


def test_endpoint_reranker_sends_the_cross_encoder_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from retrieval_eval.factories import EndpointRerankRetriever
    from tests.fixtures.factories import FixedRetriever

    calls: list[tuple[str, str, dict[str, Any]]] = []

    class FakeApi:
        def do(self, method: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
            calls.append((method, path, body))
            return {"predictions": [1.0 for _ in body["dataframe_records"]]}

    retriever = EndpointRerankRetriever(base=FixedRetriever(ids=["a"]), endpoint_name="qwen", document_chars=3)
    monkeypatch.setattr(retriever, "_client", type("W", (), {"api_client": FakeApi()})())
    retriever.invoke("query")
    assert calls == [("POST", "/serving-endpoints/qwen/invocations", {"dataframe_records": [{"query": "query", "document": "que"}]})]


def test_reranking_retriever_works_as_an_agent_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool hides the reranker: the evaluator calls one tool and gets the reranked documents back."""
    from retrieval_eval.factories import EndpointRerankRetriever, as_retriever_tool
    from tests.fixtures.factories import FixedRetriever

    monkeypatch.setattr(EndpointRerankRetriever, "score", lambda self, q, docs: [0.1, 0.9, 0.5])
    tool = as_retriever_tool(EndpointRerankRetriever(base=FixedRetriever(ids=["a", "b", "c"]), endpoint_name="qwen", top_n=2),
                             "search", "Search.")
    variant = Variant(name="v", retriever={"factory": "x.py:y"})
    t = RetrievalTarget.__new__(RetrievalTarget)
    t.variant, t.component = variant, tool
    assert [d.metadata["doc_uri"] for d in t.search("q")] == ["b", "c"]


def test_reranker_model_catalog_is_valid_and_matches_the_comparison() -> None:
    """Every custom reranker variant in rerankers.yaml points at an endpoint that reranker_models.yaml can deploy."""
    from retrieval_eval.deploy import load_models

    root = Path(__file__).parents[1] / "config"
    models = load_models(root / "reranker_models.yaml")
    assert {"qwen3_0_6b", "qwen3_4b", "gte_modernbert"} <= set(models)
    deployable = {m.endpoint_name for m in models.values()}
    config = ExperimentConfig.load(root / "rerankers.yaml")
    used = {v.retriever.params["endpoint_name"] for v in config.variants if v.retriever.factory.endswith(":endpoint_reranker")}
    assert used and used <= deployable
