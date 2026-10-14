"""MLflow "models from code" definition for a cross-encoder reranker served on Databricks Model Serving (GPU).

Ported from ai-search-product-retrieval (product_retrieval/serving/cross_encoder_model.py). Standalone on purpose: the
serving container needs only torch/transformers/sentence-transformers, not the retrieval_eval package.
Contract (matches `retrieval_eval.factories.EndpointRerankRetriever`): DataFrame[query, document] → list[float] scores.

model_config:
  family:      "cross_encoder" (sentence-transformers CrossEncoder, e.g. BAAI/bge-reranker-v2-m3,
               Alibaba-NLP/gte-reranker-modernbert-base)
               | "qwen3" (Qwen3-Reranker yes/no-logit scoring, e.g. Qwen/Qwen3-Reranker-0.6B)
  max_length:  token limit per (query, document) pair
  batch_size:  pairs per forward pass
  instruction: task instruction (qwen3 only)
artifacts:
  model_dir:   local Hugging Face snapshot of the model
"""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd

DEFAULT_INSTRUCTION = "Given a search request, judge whether the document satisfies it"
# sentence-transformers CrossEncoder models (bge-reranker, gte-reranker-modernbert, ms-marco, ...); "bge" kept as an alias
CROSS_ENCODER_FAMILIES: tuple[str, ...] = ("cross_encoder", "bge")


class CrossEncoderModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        import torch

        cfg = context.model_config or {}
        self.family = cfg.get("family", "cross_encoder")
        self.max_length = int(cfg.get("max_length", 512))
        self.batch_size = int(cfg.get("batch_size", 32))
        self.instruction = cfg.get("instruction", DEFAULT_INSTRUCTION)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = context.artifacts["model_dir"]
        if self.family in CROSS_ENCODER_FAMILIES:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(path, max_length=self.max_length, device=self.device,
                                      model_kwargs={"torch_dtype": torch.float16} if self.device == "cuda" else {})
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(self.device).eval()
            self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.no_id = self.tokenizer.convert_tokens_to_ids("no")
            prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
                      'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

    def _qwen3_scores(self, queries: list[str], documents: list[str]) -> list[float]:
        import torch

        texts = [f"<Instruct>: {self.instruction}\n<Query>: {q}\n<Document>: {d}" for q, d in zip(queries, documents)]
        budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        enc = self.tokenizer(texts, padding=False, truncation="longest_first", return_attention_mask=False, max_length=budget)
        enc["input_ids"] = [self.prefix_ids + ids + self.suffix_ids for ids in enc["input_ids"]]
        batch = self.tokenizer.pad(enc, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**batch).logits[:, -1, :]
        pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1).float()
        return torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp().tolist()

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> list[float]:
        queries, documents = model_input["query"].astype(str).tolist(), model_input["document"].astype(str).tolist()
        if self.family in CROSS_ENCODER_FAMILIES:
            return [float(s) for s in self.model.predict(list(zip(queries, documents)), batch_size=self.batch_size)]
        scores: list[float] = []
        for i in range(0, len(queries), self.batch_size):
            scores += self._qwen3_scores(queries[i:i + self.batch_size], documents[i:i + self.batch_size])
        return scores


mlflow.models.set_model(CrossEncoderModel())
