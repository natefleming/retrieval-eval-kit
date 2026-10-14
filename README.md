# retrieval-eval-kit

Measure and improve retrieval quality on **Databricks AI Search (Vector Search)** with **MLflow**.

You describe retrieval variants (embedding model, text/chunking recipe, query type, k, reranker, filters, score
threshold) in one YAML file. The kit:

1. **Generates an evaluation dataset** from the documents behind an existing index. It samples documents, writes
   synthetic questions with `databricks.agents.evals.generate_evals_df`, and stores them as a Unity Catalog
   **MLflow evaluation dataset**. Each question's ground truth is the document it was generated from.
2. **Builds the indexes the variants need.** Each one is a Delta Sync index with managed embeddings, built once per
   distinct recipe. Variants that query an existing index build nothing.
3. **Evaluates every variant** with `mlflow.genai.evaluate`, one MLflow run per variant, using these metrics:
   - deterministic, no LLM: `recall_at_k`, `hit_rate_at_k`, `precision_at_k` and `ndcg_at_k` for k = 1, 5, 10, plus
     `mrr` and `latency_ms`.
     These are document-level: chunks collapse to their `doc_uri`, so different chunking strategies are comparable.
   - LLM judges (MLflow built-ins, optional): `RetrievalRelevance`, `RetrievalSufficiency`.

```
config/experiment.yaml      settings + variants (the only file you normally edit)
config/rerankers.yaml       reranker comparison; inherits experiment.yaml's settings via `base:`
config/filters.yaml         dynamic-filter comparison, on its own constraint-style dataset
config/reranker_models.yaml custom reranker models notebook 4 can deploy
serving/                    model code for custom reranker endpoints (Qwen3-Reranker, any CrossEncoder)
src/retrieval_eval/         the code: config, dataset, indexing, factories (example retrievers), retrievers, scorers, evaluate
notebooks/01..03_*.py       one notebook per step; widgets pre-filled from the YAML
notebooks/04_deploy_rerankers.py optional: deploy custom rerankers on GPU Model Serving (scale-to-zero)
resources/*.yml             bundle resources: the evaluation job, and the optional deploy_rerankers job (serverless env 5)
databricks.yml              bundle definition: variables that override settings, targets
```

## Requirements

- Unity Catalog, serverless compute (notebooks and jobs use **environment version 5**).
- An existing **Delta Sync** AI Search index whose source table has an id column and text columns.
- `CREATE TABLE` and `CREATE SCHEMA` on the output catalog, and access to an embedding endpoint and a judge endpoint
  (defaults: `databricks-gte-large-en`, `databricks-bge-large-en`, `databricks-claude-sonnet-5-5`).

## 1. Configure

Edit `config/experiment.yaml`:

- `settings.source_index`, `doc_uri_column`, `content_columns`: the index to learn from, and how to build document
  text for question generation.
- `settings.output_schema`: where the dataset, chunk tables and new indexes go. It is created if missing. Never point
  this at the source schema.
- `settings.agent_description` / `question_guidelines`: what the system is for and what real questions look like.
  This matters most for dataset quality.
- `variants`: what to compare (see below).

## 2a. Run interactively (Git folder)

1. In the workspace: **Workspace → Create → Git folder**, and clone this repo. Or sync a local copy with
   `databricks sync . /Workspace/Users/<you>/retrieval-eval-kit`.
2. Open `notebooks/01_generate_dataset.py`, attach **serverless** (environment version 5), and **Run all**.
3. Run `notebooks/02_build_indexes.py`. Building a new index takes a while, depending on table size.
4. Run `notebooks/03_evaluate.py`. It ends with a summary table, one row per variant.

Each notebook installs the project from the folder (`%uv pip install ..`). All settings appear as widgets pre-filled
from the YAML. Change a widget to override a setting without editing the file. For example, set `variants` to
`existing_ann,existing_hybrid` to run two variants, or set `use_llm_judges` to `false` for a fast, judge-free pass.

## 2b. Deploy and run as a job (bundle)

```bash
databricks bundle validate -p <profile>
databricks bundle deploy   -p <profile>              # dev target, deployment engine: direct
databricks bundle run retrieval_eval -p <profile>    # generate_dataset + build_indexes -> evaluate
```

Override any setting:

```bash
databricks bundle deploy -p <profile> --var="output_schema=main.my_eval" --var="num_evals=200"
databricks bundle run retrieval_eval -p <profile> --params "variants=existing_hybrid existing_hybrid_rerank,use_llm_judges=false"
```

List settings (`variants`, `content_columns`) accept commas or spaces; use spaces inside `--params`, which itself
splits on commas. Variables left empty (`""`) use the YAML value. `--target prod` deploys under `/Workspace/Shared/.bundle`.

## 3. Compare results

Open the MLflow experiment. By default, interactive and job runs share `/Users/<you>/retrieval-eval-kit-experiment`
(`/Shared/retrieval-eval-kit-experiment` for `--target prod`). Override it with the `experiment_name` setting, widget or
bundle variable. The notebooks create the experiment if it is missing and tag it `mlflow.experimentKind=genai_development`,
so it is listed under **Experiments → GenAI apps & agents**.

- **Runs**: one per variant. Params include `query_type`, `embedding_model`, `chunk_size`, `rerank_columns` and `k`.
  Use the chart/compare view on `recall_at_10/mean`, `ndcg_at_10/mean`, `retrieval_relevance/mean` and
  `latency_ms/mean`.
- **Traces / evaluation tab**: per-question results with the retrieved documents (`RETRIEVER` span). Filter
  for `hit_rate_at_10 = 0` to see the misses.
- All runs from one sweep share the tag `retrieval_eval.sweep_id`.
- **Scorers tab**: `retrieval_metrics`, `retrieval_relevance` and `retrieval_sufficiency` are registered to the
  experiment the first time notebook 3 runs. From there you can run them on other traces or start them for production
  monitoring. Production traces have no ground truth, so `retrieval_metrics` reports only latency for them, while the
  judges still work. To change a registered judge's model, delete the scorer on that tab and re-run notebook 3.

## Analysis and recommendation

Notebook 3 ends with an analysis section that turns the metrics into a decision:

- **Metric overview**: variants × metrics, each column shaded from worst to best.
- **Ranking**: the `primary_metric` per variant with 95% intervals (±1.96 × standard error, from the per-question
  variance and the number of questions).
- **Recall at k**: k = 1, 5, 10. Shows whether a variant's problem is *finding* the document or *ranking* it.
- **Quality vs latency**: the cost of each gain. Each retriever is warmed up with one untraced query first, so
  cold-start time is not counted.
- **Recommendation**: the best `primary_metric` within `latency_budget_ms`. Variants within the margin of error count
  as tied and are separated by `secondary_metric`, then latency. Results are compared with `baseline_variant`, and the
  runs are tagged `retrieval_eval.role` = `recommended` / `baseline` / `candidate`.

Set `primary_metric` to how results are used: `recall_at_10` (or your k) for RAG and agents, `recall_at_1` / `mrr` /
`ndcg_at_5` when the top result matters. With small datasets most variants come out tied. The recommendation says
so, and adding questions (`num_evals`) is the fix.

## Repeatable runs

Running the job or the notebooks again gives the same setup each time:

- **Dataset**: created on the first run, then reused unchanged, so sweeps are comparable. `overwrite_dataset=true`
  deletes it and generates a new one.
- **Indexes**: built once per recipe, and existing ones are reused.
- **Experiment**: created on first use at `experiment_name`, then reused.
- **Scorers**: registered to the experiment on the first evaluation, and re-registered (as a new version) only when
  their definition changes, for example a different `judge_model`.

To hand the kit to another workspace, change only `settings` in `config/experiment.yaml`, or pass bundle variables.
Nothing else is workspace-specific.

## Defining variants

Every variant names a **retriever factory**: a function that returns the LangChain `BaseRetriever` or `BaseTool`
under test. It is called as `factory(settings, variant, **retriever.params)`. Optionally, a variant also names the
AI Search `index` it queries.

```yaml
variants:
  - name: existing_hybrid_rerank
    index: {existing: "${source_index}"}  # an index that already exists; ${setting} references a setting
    retriever:
      factory: retrieval_eval.factories:ai_search
      params:                             # keyword arguments for the factory
        query_type: HYBRID                # ANN | HYBRID | FULL_TEXT
        num_results: 10                   # default: settings.k
        rerank_columns: [product_name, description]
        filters: {brand_name NOT: "ACME"} # STANDARD endpoints: dict; STORAGE_OPTIMIZED: SQL string
        score_threshold: 0.5

  - name: chunk_512_bge                   # step 2 builds this index
    index:
      text: {columns: [title, body], separator: "\n", chunk_size: 512, chunk_overlap: 64}
      embedding_model: databricks-bge-large-en
    retriever:
      factory: retrieval_eval.factories:ai_search
      params: {query_type: HYBRID}

  - name: my_retriever                    # your own LangChain retriever or tool
    retriever:
      factory: my_retrievers.py:make_retriever   # file relative to this YAML, or package.module:function
      params: {top_k: 10}
      doc_uri_key: product_id             # result-metadata key that holds the dataset's doc_uri
```

Each factory param is logged on the MLflow run as `retriever.<name>`, so you can compare and filter runs by them.

### Example factories (`src/retrieval_eval/factories.py`)

| Factory | Returns | Use it for |
|---|---|---|
| `ai_search` | `BaseRetriever` over the AI Search Python client | Full control: query type, reranker, filters, score threshold, built indexes |
| `databricks_vector_search` | `databricks_langchain.DatabricksVectorSearch(...).as_retriever()` | The retriever most Databricks LangChain RAG apps already use |
| `retriever_tool` | that retriever wrapped by `create_retriever_tool` | Agents that call retrieval as a tool |
| `endpoint_reranker` | AI Search candidates re-scored by a custom reranker on Model Serving | Testing open-source rerankers (Qwen3-Reranker, gte-reranker-modernbert, bge-reranker, ...) |
| `endpoint_reranker_tool` | `endpoint_reranker` wrapped as a retriever tool | The same, in the form an agent calls (reranking happens inside the tool) |
| `dynamic_filter` | An LLM writes `query` + metadata `filters` per request, then searches | Agents using `VectorSearchRetrieverTool(dynamic_filter=True)` |

Reranking always lives *inside* the retriever, so the agent, and the evaluator, see one `query → documents` step.
`as_retriever_tool(retriever, name, description)` wraps any retriever, including your own, as an agent tool the same way.

### Custom reranker endpoints (`endpoint_reranker`)

`endpoint_reranker` is a two-stage retriever. AI Search returns `candidates` results (default 50). A Model Serving
endpoint scores each (question, result) pair, and the best `num_results` are returned in score order, with the score in
`metadata.rerank_score`. The reranking call appears as a `RERANKER` span in each trace, so its time is visible.

```yaml
  - name: qwen3_0_6b_rerank
    index: {existing: "${source_index}"}
    retriever:
      factory: retrieval_eval.factories:endpoint_reranker   # or endpoint_reranker_tool for the agent-tool form
      params: {endpoint_name: retrieval-qwen3-reranker-0-6b, candidates: 50, query_type: HYBRID}
```

The endpoint must accept `{"dataframe_records": [{"query": ..., "document": ...}]}` and return
`{"predictions": [score, ...]}`, one score per record.

**Deploying rerankers.** `notebooks/04_deploy_rerankers.py` deploys any of the models in
`config/reranker_models.yaml`, several at once, with the endpoints building in parallel:

| Key | Model | Hardware | Why consider it |
|---|---|---|---|
| `qwen3_0_6b` | Qwen3-Reranker-0.6B | GPU_SMALL | Good small multilingual reranker |
| `qwen3_4b` | Qwen3-Reranker-4B | GPU_MEDIUM | Highest quality in published benchmarks; slower |
| `gte_modernbert` | gte-reranker-modernbert-base (149M) | GPU_SMALL | Fastest modern English reranker |
| `bge_v2_m3` | bge-reranker-v2-m3 | GPU_SMALL | Established multilingual baseline |

Each one is downloaded from Hugging Face, registered in Unity Catalog (`output_schema`), served with
`serving/cross_encoder_model.py` on a **scale-to-zero** endpoint (no cost while idle; the first request after idle waits
for a cold start), and smoke-tested. Pick models with the `models` widget, or
`databricks bundle run deploy_rerankers -p <profile> --params "models=qwen3_4b gte_modernbert"`. To add a model, add
an entry to the file: `family: cross_encoder` works for any sentence-transformers CrossEncoder, and `family: qwen3`
for Qwen3-Reranker. Check its license first: for example, jina-reranker-v3.5 is CC BY-NC.

**Comparing rerankers.** `config/rerankers.yaml` sets `base: experiment.yaml`, so it reuses the same dataset and
settings. It compares plain hybrid search, the built-in reranker on three column choices (name; name + description;
name + brand + description), and the custom rerankers:

```bash
databricks bundle run retrieval_eval -p <profile> --params config_path=../config/rerankers.yaml
```

### Dynamic filters (`dynamic_filter`)

`VectorSearchRetrieverTool(dynamic_filter=True)` doesn't write filters itself. It adds a `filters` argument to the
tool's schema (with the index's column descriptions), and the **calling agent's LLM** fills it in. For example,
"impact driver, not DeWalt" becomes `query="impact driver"` plus `filters=[{"key": "brand_name NOT", "value": "DEWALT"}]`.
Calling the tool with only a query would never filter, so the `dynamic_filter` factory plays the agent:

1. An LLM (`llm_endpoint`) is given the real dynamic-filter tool and must call it. It writes `query` and `filters`.
2. The search runs with exactly those arguments. The plan, including the filters, is its own `plan_filters` span in
   each trace, so you can see per question what was filtered on.
3. `guarded: true` checks filter values in `guard_columns` against the column's real values (fixing casing, dropping
   values that don't exist), and searches again without filters if the filtered search returns nothing. Filters
   are exact-match, so unguarded filters with a wrong case or a guessed category silently return nothing.
4. `rerank_endpoint` optionally re-scores the results with a custom reranker (as in `endpoint_reranker`).

**Comparing.** `config/filters.yaml` compares no filters, dynamic filters, guarded dynamic filters, and guarded +
gte-modernbert reranking. It uses its **own dataset** (`retrieval_eval_dataset_constraints`), with questions that
state a brand, a category or an exclusion, because questions with nothing to filter on can't show whether filtering
helps. The first run creates it; later runs reuse it.

```bash
databricks bundle run retrieval_eval -p <profile> --params config_path=../config/filters.yaml
```

### Bring your own retriever

Copy a factory into a file next to `config/experiment.yaml` (e.g. `config/my_retrievers.py`), build your retriever or
tool in it, and add a variant that references it:

```python
from typing import Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool

from retrieval_eval.config import Settings, Variant


def make_retriever(settings: Settings, variant: Variant, **params: Any) -> BaseRetriever | BaseTool:
    from my_app.rag import build_retriever   # your existing code
    return build_retriever(index_name=settings.source_index, k=params.get("top_k", settings.k))
```

- The factory gets the shared `settings` (e.g. `source_index`, `k`) and its `variant` (e.g.
  `variant.index_name(settings)`), so one factory can serve several variants.
- **Retrievers** are called with `.invoke(query)`. **Tools** are called as a tool call with
  `{<query_arg>: query}` (`query_arg` defaults to `query`). The tool must return documents: LangChain Documents,
  dicts, JSON, or a Documents artifact (`response_format="content_and_artifact"`, as `create_retriever_tool` supports).
  A tool that returns only text loses the document ids. In that case, return its underlying retriever from the
  factory instead. This applies to `databricks_langchain.VectorSearchRetrieverTool`, whose output is a string.
- Each result's `metadata[doc_uri_key]` is the document id matched against ground truth. If it is missing, the run
  fails with the available keys listed.
- The evaluator wraps every call in one top-level `RETRIEVER` span with normalized documents. Spans that your
  retriever emits itself are kept in the trace but are not scored twice.

A new index is named `<output_schema>.rek_<hash>_index` from a hash of `text` + `embedding_model`. Changing the
recipe builds a new index, and query-only changes reuse the existing one. Delete old `rek_*` indexes and tables when
you are done.

## Tips (Databricks evaluation guidance)

- **Review the generated questions** before trusting the numbers (`display(dataset.to_df())` in notebook 1). Edit
  `question_guidelines` and regenerate with `overwrite_dataset=true` if they don't read like real user questions. Add
  real user questions to the same dataset with `merge_records` when you have them.
- **Start with the deterministic metrics.** They are free, reproducible and comparable across runs. Use the judges to
  spot-check relevance beyond the single ground-truth document. Synthetic data has one known-relevant document per
  question, so other relevant results count as misses for recall but not for `retrieval_relevance`.
- **Change one thing per variant** so each difference in metrics has one cause.
- Keep the dataset fixed while comparing variants. MLflow datasets are versioned in UC, and each run records the
  dataset it used.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest            # unit tests; real MLflow traces in a temporary local store, no workspace needed
uv run mypy src tests
```
