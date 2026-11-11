"""Stage 8: End-to-End Prefect Orchestration Workflow & Scheduled Runner."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import (
    PROJECT_ROOT,
    configure_runtime_env,
    ensure_directories,
    get_console,
    get_logger,
    load_config,
    print_stage_banner,
)

configure_runtime_env()

from prefect import flow, task
from rich.table import Table

from src.evaluation.evaluate import run_evaluation
from src.features.feature_store import run_feature_engineering
from src.ingestion.ingest import run_ingestion
from src.preparation.prepare import run_preparation
from src.training.train import run_training
from src.transformation.transform import run_transformation
from src.validation.validate import run_validation


def _timed_stage(stage_name: str, fn: Any, config_path: str | None) -> dict[str, Any]:
    t0 = time.perf_counter()
    output = fn(config_path)
    duration = round(time.perf_counter() - t0, 3)
    return {"stage": stage_name, "status": "COMPLETED", "duration_sec": duration, "result": output}


@task(name="stage-1-ingestion", retries=1, retry_delay_seconds=5)
def ingest_data_task(config_path: str | None = None) -> dict[str, Any]:
    return _timed_stage("1. Data Ingestion", run_ingestion, config_path)


@task(name="stage-2-validation", retries=0)
def validate_data_task(upstream: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    _ = upstream
    return _timed_stage("2. Data Validation & Quarantine", run_validation, config_path)


@task(name="stage-3-preparation", retries=1, retry_delay_seconds=5)
def prepare_data_task(upstream: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    _ = upstream
    return _timed_stage("3. Data Preparation & Temporal Split", run_preparation, config_path)


@task(name="stage-4-transformation", retries=1, retry_delay_seconds=5)
def transform_data_task(upstream: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    _ = upstream
    return _timed_stage("4. Data Transformation & Matrices", run_transformation, config_path)


@task(name="stage-5-feature-management", retries=1, retry_delay_seconds=5)
def feature_engineering_task(upstream: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    _ = upstream
    return _timed_stage("5. Feature Management & Catalog", run_feature_engineering, config_path)


@task(name="stage-6-model-training", retries=1, retry_delay_seconds=5)
def train_models_task(upstream: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    _ = upstream
    return _timed_stage("6. Model Training & MLflow Tracking", run_training, config_path)


@task(name="stage-7-model-evaluation", retries=1, retry_delay_seconds=5)
def evaluate_models_task(upstream: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    _ = upstream
    return _timed_stage("7. Model Evaluation & Reporting", run_evaluation, config_path)


@flow(name="recommendation-system-data-pipeline", log_prints=True)
def recommendation_pipeline_flow(config_path: str | None = None) -> dict[str, Any]:
    """Execute the complete end-to-end MovieLens recommendation data pipeline."""
    cfg = load_config(config_path)
    paths = ensure_directories(cfg)
    logger = get_logger("stage.orchestration")
    console = get_console()

    print_stage_banner(
        "Stage 8",
        "End-to-End Prefect Orchestration Flow",
        f"Flow: {cfg['orchestration']['flow_name']} | Cron: {cfg['orchestration']['cron_schedule']}",
    )

    flow_start = time.perf_counter()
    s1 = ingest_data_task(config_path=config_path)
    s2 = validate_data_task(upstream=s1, config_path=config_path)
    s3 = prepare_data_task(upstream=s2, config_path=config_path)
    s4 = transform_data_task(upstream=s3, config_path=config_path)
    s5 = feature_engineering_task(upstream=s4, config_path=config_path)
    s6 = train_models_task(upstream=s5, config_path=config_path)
    s7 = evaluate_models_task(upstream=s6, config_path=config_path)
    total_duration = round(time.perf_counter() - flow_start, 3)

    stage_records = [
        {"stage": item["stage"], "status": item["status"], "duration_sec": item["duration_sec"]}
        for item in (s1, s2, s3, s4, s5, s6, s7)
    ]

    summary = {
        "flow_name": cfg["orchestration"]["flow_name"],
        "cron_schedule": cfg["orchestration"]["cron_schedule"],
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "SUCCESS",
        "total_duration_sec": total_duration,
        "stages": stage_records,
    }
    out_path = paths["reports_dir"] / "orchestration_run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table = Table(title="Stage 8 End-to-End Orchestration Run Summary")
    table.add_column("DAG Stage", style="cyan", no_wrap=True)
    table.add_column("Task Status", style="bold green", no_wrap=True)
    table.add_column("Duration (s)", justify="right", no_wrap=True)
    for rec in stage_records:
        table.add_row(rec["stage"], rec["status"], f"{rec['duration_sec']:.3f}s")
    table.add_row("[bold]Total Pipeline[/bold]", "[bold green]SUCCESS[/bold green]", f"[bold]{total_duration:.3f}s[/bold]")
    console.print(table)
    logger.info("Orchestration summary saved to %s", out_path.relative_to(PROJECT_ROOT))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or serve the Recommendation Pipeline Prefect Flow.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to pipeline YAML config.")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Register and serve the flow on the configured cron schedule instead of running once immediately.",
    )
    args = parser.parse_args()

    if args.serve:
        cfg = load_config(args.config)
        recommendation_pipeline_flow.serve(
            name="daily-recommendation-pipeline",
            cron=cfg["orchestration"]["cron_schedule"],
        )
    else:
        recommendation_pipeline_flow(config_path=args.config)


if __name__ == "__main__":
    main()

