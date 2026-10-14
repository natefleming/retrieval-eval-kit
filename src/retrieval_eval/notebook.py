"""Notebook helpers: expose settings as widgets (pre-filled from the YAML file) and set up MLflow."""

from __future__ import annotations

from typing import Any

import mlflow
from databricks.sdk import WorkspaceClient

from retrieval_eval.config import ExperimentConfig, Settings

DEFAULT_CONFIG_PATH = "../config/experiment.yaml"


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(map(str, value))
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else str(value)


def load_config(dbutils: Any, fields: list[str]) -> ExperimentConfig:
    """Create a `config_path` widget plus one widget per setting in `fields`, then load the config with overrides.

    Widget defaults come from the YAML file, so the notebook runs as-is. Job parameters (or edited widgets) override
    them; an empty value falls back to the file.
    """
    dbutils.widgets.text("config_path", DEFAULT_CONFIG_PATH, "config_path")
    path: str = dbutils.widgets.get("config_path")
    settings: Settings = ExperimentConfig.load(path).settings
    for field in fields:
        default = _as_text(getattr(settings, field))
        if isinstance(getattr(settings, field), bool):
            dbutils.widgets.dropdown(field, default, ["true", "false"], field)
        else:
            dbutils.widgets.text(field, default, field)
    config = ExperimentConfig.load(path, {f: dbutils.widgets.get(f) for f in fields})
    print(config.settings.model_dump_json(indent=2))
    return config


def setup_mlflow(settings: Settings) -> str:
    """Use the experiment (default /Users/<me>/retrieval-eval-kit-experiment), creating it if needed.

    The experiment is tagged as a GenAI experiment so it is listed under "GenAI apps & agents" in the Experiments UI.
    """
    mlflow.set_tracking_uri("databricks")
    name = settings.experiment_name or f"/Users/{WorkspaceClient().current_user.me().user_name}/retrieval-eval-kit-experiment"
    mlflow.set_experiment(name)
    mlflow.set_experiment_tag("mlflow.experimentKind", "genai_development")
    print(f"MLflow experiment: {name}")
    return name
