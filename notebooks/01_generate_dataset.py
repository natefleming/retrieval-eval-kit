# Databricks notebook source
# MAGIC %md
# MAGIC # 1 · Generate the evaluation dataset
# MAGIC Samples `num_docs` documents from the source table behind `source_index`, generates `num_evals` synthetic shopper
# MAGIC questions with `databricks.agents.evals.generate_evals_df`, and stores them as a Unity Catalog MLflow evaluation
# MAGIC dataset (`output_schema.dataset_table`). Each record's ground truth is the `doc_uri` of the document it came from,
# MAGIC and its `source.document` records that document as the record's provenance.
# MAGIC
# MAGIC If the dataset already exists it is reused unchanged, so every sweep is scored on the same questions. Set
# MAGIC `overwrite_dataset` to `true` to delete it and generate a new one.
# MAGIC
# MAGIC Widgets are pre-filled from `config/experiment.yaml`; change a widget to override that setting.

# COMMAND ----------

# MAGIC %uv pip install -q ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from retrieval_eval.config import ExperimentConfig
from retrieval_eval.dataset import generate
from retrieval_eval.notebook import load_config, setup_mlflow

config: ExperimentConfig = load_config(
    dbutils,
    [
        "source_index", "doc_uri_column", "content_columns", "output_schema", "dataset_table", "num_docs", "num_evals",
        "agent_description", "question_guidelines", "overwrite_dataset", "experiment_name",
    ],
)
setup_mlflow(config.settings)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.settings.output_schema}")

# COMMAND ----------

dataset = generate(spark, config.settings)
display(dataset.to_df())
