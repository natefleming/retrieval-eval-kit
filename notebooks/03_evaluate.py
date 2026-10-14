# Databricks notebook source
# MAGIC %md
# MAGIC # 3 · Evaluate retrieval variants
# MAGIC
# MAGIC This notebook measures **how well each retrieval configuration finds the right documents** for a set of
# MAGIC questions, and records the results in MLflow so the configurations can be compared side by side.
# MAGIC
# MAGIC It evaluates **retrieval only**: the search step that a RAG application or agent runs before an LLM writes an
# MAGIC answer. No answers are generated here. A retriever that misses the right document limits every answer built on
# MAGIC top of it, so this is the first thing to get right.
# MAGIC
# MAGIC ## What is being evaluated
# MAGIC
# MAGIC | Concept | What it is |
# MAGIC |---|---|
# MAGIC | **Evaluation dataset** | The questions created by notebook 1, stored as an MLflow evaluation dataset in Unity Catalog. Each record has a question (`inputs.query`), the document it was generated from (`expectations.expected_retrieved_context[].doc_uri`, the ground truth), and the facts a good answer needs (`expectations.expected_facts`). |
# MAGIC | **Variant** | One retrieval configuration from `config/experiment.yaml`: a retriever factory with its parameters (e.g. query type, reranker) and, optionally, the AI Search index it queries (embedding model, text/chunking recipe). |
# MAGIC | **Run** | Each variant is evaluated in its own MLflow run, named after the variant. The run logs the variant's parameters (factory params as `retriever.*`) and the aggregate metrics below. |
# MAGIC | **Trace** | For every question, MLflow records a trace containing one `RETRIEVER` span: the query and the ranked documents returned. Scorers read these traces. Open them on the run's *Traces* / *Evaluations* tab to see per-question results. |
# MAGIC | **Scorer** | A function that grades a trace. Deterministic scorers compare retrieved `doc_uri`s with the ground truth; LLM judges read the retrieved text. All scorers are registered on the experiment's **Scorers** tab. |
# MAGIC
# MAGIC **How a question is scored:** the variant's retriever runs the question and returns a ranked list of documents.
# MAGIC Results are compared at the *document* level. Several chunks of the same document count once, at the rank of
# MAGIC the first one, so indexes with different chunking strategies compare fairly. Each metric is computed per
# MAGIC question, then averaged over the dataset (`.../mean` in MLflow).
# MAGIC
# MAGIC ## Metrics
# MAGIC
# MAGIC ### Deterministic metrics (scorer `retrieval_metrics`)
# MAGIC These need ground truth but no LLM. They are free, fast and reproducible: the same results always give the same score.
# MAGIC `k` is the **cutoff**, the number of top results considered. They are reported at k = 1, 5 and 10.
# MAGIC
# MAGIC | Metric | Question it answers | Per-question value | How to read it |
# MAGIC |---|---|---|---|
# MAGIC | `recall_at_k` | What share of the relevant documents appears in the top k? | relevant found in top k ÷ total relevant | The main coverage metric. With synthetic data each question has exactly one relevant document, so it is 1 if that document is in the top k, else 0. |
# MAGIC | `hit_rate_at_k` | Is at least one relevant document in the top k? | 1 or 0 | Equals `recall_at_k` when there is one relevant document per question. It differs once you add questions with several relevant documents. |
# MAGIC | `precision_at_k` | What share of the top k results is relevant? | relevant found in top k ÷ k | With one relevant document, the maximum is 1/k (0.1 at k=10). Compare it across variants, not against 1.0. |
# MAGIC | `ndcg_at_k` | Were the relevant documents ranked near the top? | DCG ÷ ideal DCG (0 to 1) | Rewards finding the document *and* ranking it high: rank 1 = 1.0, rank 2 ≈ 0.63, rank 3 = 0.5, rank 10 ≈ 0.29, missed = 0. |
# MAGIC | `mrr` | How far down is the first relevant document? | 1 ÷ rank of first relevant result | Rank 1 = 1.0, rank 2 = 0.5, rank 3 = 0.33. Computed over all results the retriever returns, not only the top 10. |
# MAGIC | `search_time_ms` | How long does one search take? | milliseconds | Measured after the evaluation on `search_time_samples` questions (default 20), **one at a time** after a warm-up, including the network round trip and any reranking. This is what the report calls *search time*. Use it to compare variants, not as a production SLA. |
# MAGIC | `latency_ms` | How long did searches take during the evaluation? | milliseconds per query | The evaluation sends many questions **in parallel**, so this includes queueing. It is much higher than `search_time_ms` for endpoints that serve one request at a time, such as a single GPU reranker replica. Raw metrics only. |
# MAGIC
# MAGIC ### LLM-judged metrics (MLflow built-in judges, on when `use_llm_judges` is `true`)
# MAGIC An LLM (`judge_model`) reads the retrieved text. Judges need no exact ground-truth match, so they also credit relevant
# MAGIC documents that are not the one the question was generated from. They cost tokens and vary slightly between runs.
# MAGIC
# MAGIC | Metric | Question it answers | How to read it |
# MAGIC |---|---|---|
# MAGIC | `retrieval_relevance` | Is each retrieved chunk relevant to the question? | Share of retrieved chunks judged relevant (0 to 1). High with low recall means plausible alternatives were returned, not the exact document. |
# MAGIC | `retrieval_sufficiency` | Do the retrieved chunks contain everything needed to answer (the `expected_facts`)? | Share of questions where the context was judged sufficient (0 to 1). The closest proxy for "could an LLM answer correctly from this?". |
# MAGIC
# MAGIC ## Acronyms and terms
# MAGIC
# MAGIC | Term | Meaning |
# MAGIC |---|---|
# MAGIC | **IR** | Information retrieval: finding the documents relevant to a query. Recall, precision, nDCG and MRR are standard IR metrics. |
# MAGIC | **RAG** | Retrieval-augmented generation: an LLM answers using documents retrieved for the question. |
# MAGIC | **k / @k** | The cutoff: only the top k results are considered (e.g. `recall_at_5` = recall@5). |
# MAGIC | **nDCG** | Normalized discounted cumulative gain. **DCG** adds 1 ÷ log₂(rank + 1) for every relevant result, so lower ranks count less. Dividing by the best possible DCG (**ideal DCG**) scales it to 0 to 1. |
# MAGIC | **MRR** | Mean reciprocal rank: the average over questions of 1 ÷ (rank of the first relevant result). |
# MAGIC | **ANN** | Approximate nearest neighbor: vector (semantic) search over embeddings. Finds meaning, can miss exact terms such as model numbers. |
# MAGIC | **FULL_TEXT / BM25** | Keyword search ranked with BM25 (Okapi Best Match 25), a term-frequency scoring formula. Finds exact terms, misses paraphrases. |
# MAGIC | **HYBRID** | ANN and keyword results merged, usually with **RRF** (reciprocal rank fusion: each list contributes 1 ÷ (constant + rank)). |
# MAGIC | **Reranker** | A second model that re-scores the retrieved candidates against the query (`rerank_columns`). It usually improves the top ranks at some latency cost. |
# MAGIC | **Embedding model** | The model that turns text into vectors for ANN search (e.g. `databricks-gte-large-en`, `databricks-bge-large-en`). |
# MAGIC | **Chunk** | A piece of a document that is indexed and retrieved on its own. **Chunk size / overlap** control how documents are split. |
# MAGIC | **doc_uri** | The document identifier used as ground truth (here the product id). Retrieved results are matched on it. |
# MAGIC | **Ground truth** | The known correct answer for a question. Here, the document the question was generated from. |
# MAGIC | **LLM judge** | An LLM prompted to grade output against a rubric. Here, MLflow's built-in retrieval judges. |
# MAGIC | **UC** | Unity Catalog, where the evaluation dataset, chunk tables and indexes are stored and governed. |
# MAGIC
# MAGIC ## Worked example
# MAGIC Suppose the right product comes back at **rank 3** of 10 results:
# MAGIC `recall_at_1` = 0, `recall_at_5` = 1, `recall_at_10` = 1, `precision_at_5` = 0.2, `ndcg_at_5` = 1 ÷ log₂(4) = 0.5,
# MAGIC `mrr` = 1/3. Averaged over all questions, `recall_at_10` = 0.95 with `recall_at_1` = 0.67 means the right document
# MAGIC is almost always found but ranked first only two-thirds of the time. That gap is what a reranker or a better
# MAGIC query type should close.
# MAGIC
# MAGIC ## Choosing what to optimize
# MAGIC - **An LLM reads all the results (RAG, agents):** prioritize `recall_at_k` for the k you pass to the LLM, and `retrieval_sufficiency`.
# MAGIC - **A person or agent uses the first result (search box, "best match"):** prioritize `recall_at_1`, `mrr` and `ndcg_at_5`.
# MAGIC - **Always** check `latency_ms` against your budget, and look at a few misses in the traces before trusting a number.
# MAGIC
# MAGIC ## Caveats
# MAGIC - **Synthetic questions have one known-relevant document.** Other genuinely relevant results (e.g. a near-identical
# MAGIC   product) count as misses for the deterministic metrics but not for the judges. Read both kinds together.
# MAGIC - **Small datasets are noisy.** With 60 questions, one question is about 0.017 of recall, so differences under ~0.05
# MAGIC   may be noise. Use 100–200+ questions before choosing a configuration.
# MAGIC - **Compare variants on the same dataset.** Runs record the dataset they used (`retrieval_eval.dataset` tag). Runs
# MAGIC   from one execution share a `retrieval_eval.sweep_id` tag.

# COMMAND ----------

# MAGIC %uv pip install -q ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd

from retrieval_eval.config import ExperimentConfig
from retrieval_eval.evaluate import run
from retrieval_eval.notebook import load_config, setup_mlflow

config: ExperimentConfig = load_config(
    dbutils,
    [
        "source_index", "doc_uri_column", "content_columns", "output_schema", "dataset_table", "k", "judge_model",
        "use_llm_judges", "experiment_name", "variants", "primary_metric", "secondary_metric", "latency_budget_ms",
        "baseline_variant",
    ],
)
experiment: str = setup_mlflow(config.settings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the evaluation
# MAGIC Each configuration answers every test question; this takes a few minutes per configuration (longer with the AI
# MAGIC judges on). Results are logged to MLflow as they finish. The analysis below explains them.

# COMMAND ----------

summary: pd.DataFrame = run(config, experiment)
print(f"Evaluated {len(summary)} configurations. MLflow experiment: {experiment}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results and recommendation
# MAGIC
# MAGIC ### How to read these results
# MAGIC Every test question was written from one specific document, the **correct document** for that question. Each
# MAGIC configuration searched for every question, and we checked where the correct document appeared in its results:
# MAGIC
# MAGIC | Measure | What it tells you | Good looks like |
# MAGIC |---|---|---|
# MAGIC | **Found in top 10** (recall@10) | How often the correct document is anywhere in the first 10 results. The key number when an AI assistant reads all the results to write its answer. | Close to 100% |
# MAGIC | **Found in top 5** (recall@5) | The same, for the first 5 results. | Close to the top-10 number |
# MAGIC | **Ranked first** (recall@1) | How often the correct document is the very first result. The key number when people (or an app) only look at the top result. | As high as possible; usually well below top 10 |
# MAGIC | **Ranking score** (nDCG@10) | One number for *how high* the correct document is ranked: 1.00 = always first, 0.50 = typically third, 0 = not in the top 10. | Higher is better |
# MAGIC | **Position score** (MRR) | 1 ÷ the position of the correct document, averaged: 1.00 = always #1, 0.50 = typically #2. | Higher is better |
# MAGIC | **Relevant results** (AI judge) | Share of all returned results that an AI model judges relevant to the question. It also credits good results that are not the specific correct document. | Higher is better; naturally lower than "found in top 10" |
# MAGIC | **Enough to answer** (AI judge) | How often the results contain all the facts needed to answer the question. | Close to 100% |
# MAGIC | **Search time** | Average time for one search, in milliseconds, measured one question at a time. | Within your application's budget |
# MAGIC
# MAGIC **Margin of error.** Each number comes from a limited set of test questions, so it is an estimate. Charts show a
# MAGIC *likely range* (the 95% margin of error) as whiskers. When two configurations' ranges overlap a lot, the test
# MAGIC can't tell them apart yet. Adding more questions narrows the ranges.
# MAGIC
# MAGIC **Colors.** In the charts, **blue** is the recommended configuration, **orange** is today's setup (`baseline_variant`),
# MAGIC and **gray** shows the others (hover for names). In tables, ★ marks the recommended configuration and ◆ today's setup.
# MAGIC
# MAGIC **How the recommendation is chosen.** The configuration with the best **primary measure** (`primary_metric`,
# MAGIC by default *found in top 10*) wins, among those within `latency_budget_ms` if you set one. Configurations too close
# MAGIC to call are separated by the **secondary measure** (`secondary_metric`, by default *ranking score*), then by speed.
# MAGIC For search experiences where only the first result matters, set `primary_metric` to `recall_at_1`.

# COMMAND ----------

import mlflow.genai.datasets

from retrieval_eval.analysis import (
    Recommendation, analyze, chart_html, key_findings_html, quality_latency_chart, recall_curve_chart, ranking_chart,
    recommendation_html, results_table, tag_runs, time_metric,
)

num_questions: int = len(mlflow.genai.datasets.get_dataset(name=config.settings.dataset_name).to_df())
table: pd.DataFrame
recommendation: Recommendation
table, recommendation = analyze(config, summary, num_questions)
tag_runs(summary, recommendation)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key findings

# COMMAND ----------

displayHTML(key_findings_html(table, recommendation, config))

# COMMAND ----------

# MAGIC %md
# MAGIC ### All configurations at a glance
# MAGIC One row per configuration. Each column is shaded from the worst result (light) to the best (dark blue), so compare
# MAGIC down a column. For search time, faster is darker.

# COMMAND ----------

displayHTML(results_table(table, recommendation, config).to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Which configuration does best on the primary measure?
# MAGIC Longer bars are better. The whisker on each bar is its likely range. If the recommended configuration's whisker
# MAGIC overlaps the end of today's setup's bar, the improvement is not yet certain.

# COMMAND ----------

displayHTML(chart_html(ranking_chart(table, recommendation)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Finding the correct document vs. ranking it first
# MAGIC Each line shows how often the correct document is in the top 1, top 5 and top 10 results.
# MAGIC - A line that **starts low and rises steeply** finds the right document but ranks it too low. A reranker or
# MAGIC   hybrid search usually helps.
# MAGIC - A line that **stays low** doesn't find the right document at all. Look at the embedding model, the text that
# MAGIC   is indexed, or how documents are split (chunking).

# COMMAND ----------

displayHTML(chart_html(recall_curve_chart(table, recommendation)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Better results vs. speed
# MAGIC Each dot is a configuration. **Up** means better results and **left** means faster, so the top-left corner is
# MAGIC best. If one dot is both higher and further left than another, it is better on both counts. Rerankers usually
# MAGIC move a configuration up (better) and to the right (slower).

# COMMAND ----------

if time_metric(table) in table:
    displayHTML(chart_html(quality_latency_chart(table, recommendation)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recommendation

# COMMAND ----------

recommended_variant = next(v for v in config.variants if v.name == recommendation.variant)
recommended_params: dict[str, str] = mlflow.get_run(summary.at[recommendation.variant, "run_id"]).data.params
displayHTML(recommendation_html(recommendation, table, recommended_variant, recommended_params))

# COMMAND ----------

# MAGIC %md
# MAGIC ### For technical users: all raw metrics
# MAGIC Every aggregate MLflow logged, per configuration (`<metric>/mean`, `<metric>/variance`, and the MLflow `run_id`).
# MAGIC The same numbers are on each run in the experiment UI.

# COMMAND ----------

display(summary.reset_index())
