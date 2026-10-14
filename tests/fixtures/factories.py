"""Factories used by the tests; loaded both by module path and by file path."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool

from retrieval_eval.config import Settings, Variant


class FixedRetriever(BaseRetriever):
    ids: list[str | float]
    key: str = "doc_uri"

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [Document(page_content=f"{query} {i}", metadata={self.key: i}) for i in self.ids]


def fixed(settings: Settings, variant: Variant, *, ids: list[str | float], key: str = "doc_uri") -> BaseRetriever:
    return FixedRetriever(ids=ids, key=key)


def artifact_tool(settings: Settings, variant: Variant, *, ids: list[str]) -> BaseTool:
    def search(query: str) -> tuple[str, list[Document]]:
        docs = [Document(page_content=i, metadata={"doc_uri": i}) for i in ids]
        return "\n".join(d.page_content for d in docs), docs

    return StructuredTool.from_function(search, name="search", description="search", response_format="content_and_artifact")


def json_tool(settings: Settings, variant: Variant, *, ids: list[str]) -> BaseTool:
    def search(query: str) -> str:
        return json.dumps([{"page_content": i, "metadata": {"doc_uri": i}} for i in ids])

    return StructuredTool.from_function(search, name="search", description="search")


def text_tool(settings: Settings, variant: Variant, **params: Any) -> BaseTool:
    def search(query: str) -> str:
        return "just some text"

    return StructuredTool.from_function(search, name="search", description="search")


def not_a_retriever(settings: Settings, variant: Variant) -> object:
    return object()
