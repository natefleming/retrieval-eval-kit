"""Deploy custom reranker models (cross-encoders) on Model Serving, for the `endpoint_reranker` factory.

The models to deploy are listed in `config/reranker_models.yaml`. Each is downloaded from Hugging Face, registered in
Unity Catalog with the model code in `serving/cross_encoder_model.py`, and served on a scale-to-zero endpoint with the
contract `{"dataframe_records": [{"query", "document"}]}` → `{"predictions": [score, ...]}`.

Heavy dependencies (huggingface_hub, torch, transformers) are imported inside the functions that need them, so the
rest of the package does not require them.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import time
from pathlib import Path
from typing import Any, Literal

import mlflow
import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType
from pydantic import BaseModel, Field

MODEL_CODE: str = "serving/cross_encoder_model.py"  # relative to the project root
EXAMPLE: dict[str, list[str]] = {"query": ["cordless drill"], "document": ["20V MAX cordless drill/driver kit"]}


class RerankerModel(BaseModel):
    """One reranker to deploy."""

    hf_repo: str = Field(description="Hugging Face model id, e.g. Qwen/Qwen3-Reranker-0.6B")
    family: Literal["qwen3", "cross_encoder"] = Field(
        description="qwen3 = Qwen3-Reranker yes/no scoring; cross_encoder = sentence-transformers CrossEncoder (bge, gte, ...)"
    )
    endpoint_name: str = Field(description="Model Serving endpoint to create or update")
    model_name: str = Field(description="Registered model name inside output_schema")
    workload_type: Literal["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE", "CPU"] = Field("GPU_SMALL", description="Serving hardware")
    max_length: int = Field(512, description="Token limit per (query, document) pair")
    batch_size: int = Field(32, description="Pairs scored per forward pass")
    instruction: str = Field(
        "Given a search request, judge whether the document satisfies it", description="Task instruction (qwen3 only)"
    )
    description: str = Field("", description="Shown when listing models")


def load_models(path: str | Path) -> dict[str, RerankerModel]:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return {key: RerankerModel.model_validate(spec) for key, spec in (raw.get("models") or {}).items()}


def register(model: RerankerModel, output_schema: str, project_root: str | Path) -> tuple[str, str]:
    """Download the model, log it with the cross-encoder model code, and register it in UC. Returns (name, version)."""
    from huggingface_hub import snapshot_download

    mlflow.set_registry_uri("databricks-uc")
    uc_name: str = f"{output_schema}.{model.model_name}"
    local_dir: str = snapshot_download(model.hf_repo, local_dir=f"/tmp/hf/{model.hf_repo.split('/')[-1]}")
    requirements: list[str] = [
        f"torch=={md.version('torch').split('+')[0]}", f"transformers=={md.version('transformers')}", "accelerate",
        f"mlflow=={md.version('mlflow')}", "pandas",
    ] + ([f"sentence-transformers=={md.version('sentence-transformers')}"] if model.family == "cross_encoder" else [])
    model_config: dict[str, Any] = {
        "family": model.family, "max_length": model.max_length, "batch_size": model.batch_size,
        "instruction": model.instruction,
    }
    import pandas as pd
    from mlflow.models import infer_signature

    # An explicit signature: with only an input example, MLflow loads the model and runs it on this (CPU) compute to
    # infer one, which runs out of memory for multi-billion-parameter rerankers. The model only loads on the endpoint.
    signature = infer_signature(pd.DataFrame(EXAMPLE), [0.5])
    with mlflow.start_run(run_name=f"register_{model.endpoint_name}"):
        mlflow.set_tags({"retrieval_eval.purpose": "reranker_registration", "retrieval_eval.hf_repo": model.hf_repo})
        info = mlflow.pyfunc.log_model(
            name="model", python_model=os.path.abspath(Path(project_root) / MODEL_CODE),
            artifacts={"model_dir": local_dir}, model_config=model_config, pip_requirements=requirements,
            signature=signature, registered_model_name=uc_name,
        )
    return uc_name, str(info.registered_model_version)


def serve(w: WorkspaceClient, model: RerankerModel, uc_name: str, version: str) -> None:
    """Create the endpoint, or point an existing one at the new model version (scale-to-zero)."""
    entity: ServedEntityInput = ServedEntityInput(
        entity_name=uc_name, entity_version=version, workload_type=ServingModelWorkloadType(model.workload_type),
        workload_size="Small", scale_to_zero_enabled=True,
    )
    if model.endpoint_name in {e.name for e in w.serving_endpoints.list()}:
        w.serving_endpoints.update_config(name=model.endpoint_name, served_entities=[entity])
    else:
        w.serving_endpoints.create(
            name=model.endpoint_name, config=EndpointCoreConfigInput(name=model.endpoint_name, served_entities=[entity])
        )


def wait_until_ready(w: WorkspaceClient, endpoints: list[str], timeout_s: int = 3 * 3600, poll_s: int = 60) -> None:
    """Wait for all endpoints to be READY with no pending update; raise on the first failed build."""
    pending: set[str] = set(endpoints)
    deadline: float = time.time() + timeout_s
    while pending:
        for name in sorted(pending):
            state = w.serving_endpoints.get(name).state
            ready, update = str(state.ready if state else ""), str(state.config_update if state else "")
            if ready.endswith("READY") and update.endswith("NOT_UPDATING"):
                pending.discard(name)
                print(f"{name}: ready")
            elif update.endswith("UPDATE_FAILED"):
                raise RuntimeError(f"{name} failed to deploy; see its build logs in Serving: {state}")
        if pending:
            if time.time() > deadline:
                raise TimeoutError(f"not ready after {timeout_s}s: {sorted(pending)}")
            print(f"waiting for: {', '.join(sorted(pending))}")
            time.sleep(poll_s)


def smoke_test(w: WorkspaceClient, endpoint_name: str) -> tuple[float, float]:
    """Score one relevant and one unrelated document; the relevant one must score higher."""
    response: Any = w.api_client.do("POST", f"/serving-endpoints/{endpoint_name}/invocations", body={"dataframe_records": [
        {"query": "cordless drill", "document": "20V MAX cordless drill/driver kit with two batteries"},
        {"query": "cordless drill", "document": "50 ft. heavy-duty garden hose"},
    ]})
    relevant, unrelated = (float(s) for s in response["predictions"])
    if relevant <= unrelated:
        raise AssertionError(f"{endpoint_name} did not prefer the relevant document ({relevant:.3f} vs {unrelated:.3f})")
    return relevant, unrelated
