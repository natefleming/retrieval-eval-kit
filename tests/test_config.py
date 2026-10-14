from pathlib import Path

import pytest
from pydantic import ValidationError

from retrieval_eval.config import ExperimentConfig, IndexConfig, TextConfig, Variant

CONFIG = Path(__file__).parents[1] / "config" / "experiment.yaml"


def index_of(variant: Variant) -> IndexConfig:
    assert variant.index is not None
    return variant.index


def test_example_config_loads_and_resolves_references() -> None:
    config = ExperimentConfig.load(CONFIG)
    existing = next(v for v in config.variants if v.name == "existing_ann")
    assert index_of(existing).existing == config.settings.source_index


def test_empty_overrides_fall_back_to_file() -> None:
    config = ExperimentConfig.load(CONFIG, {"k": "", "num_evals": "25", "variants": "existing_ann existing_hybrid"})
    assert config.settings.k == 10
    assert config.settings.num_evals == 25
    assert [v.name for v in config.selected_variants()] == ["existing_ann", "existing_hybrid"]


def test_override_flows_into_variant_references() -> None:
    config = ExperimentConfig.load(CONFIG, {"source_index": "c.s.other_index"})
    assert index_of(config.variants[0]).existing == "c.s.other_index"


def test_unknown_variant_rejected() -> None:
    with pytest.raises(ValueError, match="unknown variants"):
        ExperimentConfig.load(CONFIG, {"variants": "nope"}).selected_variants()


def test_index_needs_exactly_one_kind() -> None:
    with pytest.raises(ValidationError):
        IndexConfig()
    with pytest.raises(ValidationError):
        IndexConfig(existing="c.s.i", text=TextConfig(columns=["a"]), embedding_model="m")


def test_same_recipe_shares_an_index() -> None:
    config = ExperimentConfig.load(CONFIG)
    gte, bge = (next(v for v in config.variants if v.name == n) for n in ("composite_gte_hybrid", "composite_bge_hybrid"))
    same = IndexConfig(text=index_of(gte).text, embedding_model=index_of(gte).embedding_model)
    assert same.key == index_of(gte).key != index_of(bge).key
    assert gte.index_name(config.settings) == f"{config.settings.output_schema}.rek_{index_of(gte).key}_index"


def test_every_variant_names_a_factory() -> None:
    with pytest.raises(ValidationError, match="retriever"):
        Variant(name="v")  # type: ignore[call-arg]  # deliberately missing `retriever`


def test_index_is_optional_and_factory_params_are_logged() -> None:
    variant = Variant(name="v", retriever={"factory": "my.py:make", "params": {"query_type": "HYBRID", "cols": ["a"]}})
    config = ExperimentConfig.load(CONFIG)
    assert variant.index_name(config.settings) is None
    params = variant.params(config.settings)
    assert params["retriever"] == "my.py:make"
    assert params["retriever.query_type"] == "HYBRID" and params["retriever.cols"] == '["a"]'
    assert config.base_dir == CONFIG.parent.resolve()


def test_params_are_readable_and_never_null() -> None:
    config = ExperimentConfig.load(CONFIG)
    by_name = {v.name: v for v in config.variants}
    existing = by_name["existing_ann"].params(config.settings, {"embedding_model": "databricks-gte-large-en", "embedded_text": "description"})
    assert existing["index_source"] == "existing" and existing["embedding_model"] == "databricks-gte-large-en"
    assert existing["embedded_text"] == "description" and existing["chunking"] == "as built (existing index)"
    built = by_name["composite_bge_hybrid"].params(config.settings)
    assert built["index_source"] == "built" and built["embedding_model"] == "databricks-bge-large-en"
    assert built["embedded_text"] == "product_name + brand_name + description" and built["chunking"] == "whole document"
    chunked = Variant(name="c", index={"text": {"columns": ["d"], "chunk_size": 512, "chunk_overlap": 64}, "embedding_model": "m"},
                      retriever={"factory": "f.py:f"}).params(config.settings)
    assert chunked["chunking"] == "512 chars, 64 overlap" and chunked["chunk_size"] == 512 and chunked["chunk_overlap"] == 64
    for params in (existing, built, chunked):
        assert None not in params.values() and "existing" != params["embedding_model"]


def test_base_file_inheritance() -> None:
    rerankers = ExperimentConfig.load(CONFIG.parent / "rerankers.yaml")
    main = ExperimentConfig.load(CONFIG)
    assert rerankers.settings.dataset_name == main.settings.dataset_name  # same dataset → comparable results
    assert rerankers.settings.baseline_variant == "hybrid" and main.settings.baseline_variant == "existing_ann"
    assert [v.name for v in rerankers.variants][:4] == [
        "hybrid", "builtin_rerank_name", "builtin_rerank_name_desc", "builtin_rerank_name_brand_desc"]
    assert index_of(rerankers.variants[0]).existing == main.settings.source_index  # ${source_index} from the base file
    assert ExperimentConfig.load(CONFIG.parent / "rerankers.yaml", {"k": "5"}).settings.k == 5
