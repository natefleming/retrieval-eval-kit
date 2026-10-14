# Databricks notebook source
# MAGIC %md
# MAGIC # 4 · Deploy custom reranker models (optional)
# MAGIC
# MAGIC A **reranker** re-scores the results a search returns, putting the most relevant first. AI Search has a built-in
# MAGIC reranker (`ai_search` with `rerank_columns`). This notebook deploys **open-source rerankers of your choice** on
# MAGIC Model Serving, so the `endpoint_reranker` factory can test whether any of them beats the built-in one on your data.
# MAGIC
# MAGIC The available models are listed in `config/reranker_models.yaml`:
# MAGIC
# MAGIC | Key | Model | Why consider it |
# MAGIC |---|---|---|
# MAGIC | `qwen3_0_6b` | Qwen3-Reranker-0.6B | Good small multilingual reranker |
# MAGIC | `qwen3_4b` | Qwen3-Reranker-4B | Highest quality in published benchmarks; slower, larger GPU |
# MAGIC | `gte_modernbert` | gte-reranker-modernbert-base (149M) | Fastest modern English reranker |
# MAGIC | `bge_v2_m3` | bge-reranker-v2-m3 | Established multilingual baseline |
# MAGIC
# MAGIC To add another model, add an entry to that file. Use `family: cross_encoder` for any sentence-transformers
# MAGIC CrossEncoder, or `qwen3` for Qwen3-Reranker. Check the license first.
# MAGIC
# MAGIC For each selected model, the notebook downloads it from Hugging Face, registers it in Unity Catalog
# MAGIC (`output_schema`), creates or updates a **scale-to-zero** serving endpoint (no cost while idle), waits until all
# MAGIC endpoints are ready (GPU builds usually take 15–40 minutes, and they build in parallel), and sends a test request.
# MAGIC Re-running registers a new model version and updates the endpoint.
# MAGIC
# MAGIC Then compare them: set the `config_path` widget in notebook 3 to `../config/rerankers.yaml`, or run
# MAGIC `databricks bundle run retrieval_eval --params config_path=../config/rerankers.yaml`.

# COMMAND ----------

# MAGIC %uv pip install -q .. huggingface_hub "transformers>=4.51" "sentence-transformers>=3.4" torch

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient

from retrieval_eval.config import ExperimentConfig
from retrieval_eval.deploy import RerankerModel, load_models, register, serve, smoke_test, wait_until_ready
from retrieval_eval.notebook import load_config

config: ExperimentConfig = load_config(dbutils, ["output_schema"])
dbutils.widgets.text("models_file", "../config/reranker_models.yaml", "models_file")
available: dict[str, RerankerModel] = load_models(dbutils.widgets.get("models_file"))
dbutils.widgets.text("models", "qwen3_0_6b", f"models (any of: {', '.join(available)})")

selected: list[str] = [m.strip() for m in dbutils.widgets.get("models").replace(",", " ").split() if m.strip()]
unknown: list[str] = [m for m in selected if m not in available]
if unknown:
    raise ValueError(f"unknown models {unknown}; choose from {list(available)}")
w: WorkspaceClient = WorkspaceClient()
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.settings.output_schema}")
for key in selected:
    m = available[key]
    print(f"{key}: {m.hf_repo} ({m.family}) → {config.settings.output_schema}.{m.model_name} → {m.endpoint_name} [{m.workload_type}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the models and start the endpoint builds

# COMMAND ----------

for key in selected:
    model: RerankerModel = available[key]
    uc_name, version = register(model, config.settings.output_schema, project_root="..")
    serve(w, model, uc_name, version)
    print(f"{key}: registered {uc_name} v{version}; endpoint {model.endpoint_name} building")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait until every endpoint is ready, then test each one
# MAGIC The relevant document should score clearly higher than the unrelated one.

# COMMAND ----------

wait_until_ready(w, [available[k].endpoint_name for k in selected])
for key in selected:
    relevant, unrelated = smoke_test(w, available[key].endpoint_name)
    print(f"{key}: relevant {relevant:.3f}  unrelated {unrelated:.3f}")
