"""Configuration loader, path resolution, and structured console logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline_config.yaml"

_CONSOLE = Console(
    width=int(os.environ.get("COLUMNS", "118")),
    force_terminal=True if os.environ.get("FORCE_COLOR") == "1" else None,
    soft_wrap=True,
)


def configure_runtime_env() -> None:
    """Silence noisy telemetry/hints from Prefect and MLflow during pipeline runs."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    os.environ.setdefault("DO_NOT_TRACK", "1")


def get_console() -> Console:
    """Return shared Rich console instance."""
    return _CONSOLE


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a structured Rich-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=_CONSOLE,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%H:%M:%S]"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML pipeline configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(cfg: dict[str, Any], path_key: str) -> Path:
    """Resolve a configured relative path against the repository root."""
    raw = Path(cfg["paths"][path_key])
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def ensure_directories(cfg: dict[str, Any]) -> dict[str, Path]:
    """Create all configured pipeline directories if absent and return resolved paths."""
    resolved: dict[str, Path] = {}
    for key in cfg["paths"]:
        directory = resolve_path(cfg, key)
        directory.mkdir(parents=True, exist_ok=True)
        resolved[key] = directory
    (resolved["reports_dir"] / "screenshots").mkdir(parents=True, exist_ok=True)
    return resolved


def print_stage_banner(stage: str, title: str, subtitle: str = "") -> None:
    """Render a consistent header banner for pipeline stage CLI output."""
    text = f"[bold cyan]{stage}: {title}[/bold cyan]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"
    _CONSOLE.print(Panel(text, border_style="cyan", expand=False))
