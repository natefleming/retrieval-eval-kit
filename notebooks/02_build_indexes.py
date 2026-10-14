# Databricks notebook source
# MAGIC %md
# MAGIC # 2 · Build variant indexes
# MAGIC For every variant in the experiment file that defines `index.text` + `index.embedding_model`, writes a chunk table
# MAGIC and creates a Delta Sync index with managed embeddings in `output_schema`. Variants that query an existing index
# MAGIC are skipped, and variants sharing a recipe share one index. Re-running only builds what is missing.

# COMMAND ----------

# MAGIC %uv pip install -q ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from retrieval_eval.config import ExperimentConfig
from retrieval_eval.indexing import build_all
from retrieval_eval.notebook import load_config

config: ExperimentConfig = load_config(
    dbutils, ["source_index", "doc_uri_column", "output_schema", "vs_endpoint", "variants"]
)

# COMMAND ----------

indexes: list[str] = build_all(spark, config)
print("Indexes ready:", indexes or "none needed")
