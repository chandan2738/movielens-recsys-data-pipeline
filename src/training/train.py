"""Stage 6: Model Training & MLflow Experiment Tracking."""

from __future__ import annotations

import json
import pickle
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

import mlflow
import pandas as pd
from rich.table import Table
from scipy import sparse

from src.features.feature_store import FeatureStore
from src.training.models import HybridSVDRecommender, PopularityRecommender


def _resolve_mlflow_uri(raw_uri: str) -> str:
    if raw_uri.startswith("sqlite:///"):
        rel = raw_uri.removeprefix("sqlite:///")
        path = Path(rel)
        abs_path = path if path.is_absolute() else PROJECT_ROOT / path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{abs_path}"
    return raw_uri


def run_training(config_path: str | Path | None = None) -> dict[str, Any]:
    """Train Baseline Popularity and Hybrid Personalized recommenders and track in MLflow."""
    configure_runtime_env()
    cfg = load_config(config_path)
    paths = ensure_directories(cfg)
    logger = get_logger("stage.training")
    console = get_console()

    print_stage_banner(
        "Stage 6",
        "Model Training & MLflow Tracking",
        f"Experiment: {cfg['training']['mlflow_experiment_name']} | URI: {cfg['training']['mlflow_tracking_uri']}",
    )

    proc_dir = paths["processed_dir"]
    feat_dir = paths["features_dir"]
    models_dir = paths["models_dir"]

    store = FeatureStore.load(feat_dir)
    _, item_features, _ = store.get_training_features()
    train_df = pd.read_parquet(proc_dir / "train_interactions.parquet")
    rating_mat = sparse.load_npz(proc_dir / "user_item_rating_matrix.npz")
    implicit_mat = sparse.load_npz(proc_dir / "user_item_implicit_matrix.npz")
    mappings = json.loads((proc_dir / "id_mappings.json").read_text(encoding="utf-8"))

    tracking_uri = _resolve_mlflow_uri(cfg["training"]["mlflow_tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(cfg["training"]["mlflow_experiment_name"])

    # 1. Baseline Popularity Recommender
    pop_cfg = cfg["training"]["popularity_model"]
    t0 = time.perf_counter()
    pop_model = PopularityRecommender(
        bayesian_m=float(pop_cfg["bayesian_m"]),
        popularity_weight=float(pop_cfg["popularity_weight"]),
    ).fit(item_features=item_features, train_df=train_df)
    pop_time = time.perf_counter() - t0

    pop_path = models_dir / "popularity_recommender.pkl"
    with pop_path.open("wb") as handle:
        pickle.dump(pop_model, handle)

    with mlflow.start_run(run_name="baseline_popularity") as pop_run:
        mlflow.set_tags({"model_family": "baseline", "personalized": "false", "feature_store_version": store.metadata["feature_store_version"]})
        mlflow.log_params(
            {
                "model_type": "PopularityRecommender",
                "bayesian_m": pop_cfg["bayesian_m"],
                "popularity_weight": pop_cfg["popularity_weight"],
                "train_rows": len(train_df),
                "n_items": mappings["n_items"],
            }
        )
        mlflow.log_metric("train_duration_sec", round(pop_time, 4))
        mlflow.log_artifact(str(pop_path), artifact_path="model_artifacts")
        mlflow.log_artifact(str(feat_dir / "feature_metadata.json"), artifact_path="feature_store")
        pop_run_id = pop_run.info.run_id

    # 2. Personalized Hybrid ALS + Implicit SVD + Content Genre Recommender
    hyb_cfg = cfg["training"]["hybrid_model"]
    t0 = time.perf_counter()
    hybrid_model = HybridSVDRecommender(
        n_factors=int(hyb_cfg["n_factors"]),
        n_epochs=int(hyb_cfg["n_epochs"]),
        reg=float(hyb_cfg["reg"]),
        bias_reg=float(hyb_cfg["bias_reg"]),
        implicit_svd_factors=int(hyb_cfg["implicit_svd_factors"]),
        explicit_weight=float(hyb_cfg["explicit_weight"]),
        implicit_weight=float(hyb_cfg["implicit_weight"]),
        genre_weight=float(hyb_cfg["genre_weight"]),
        random_state=int(cfg["project"]["random_seed"]),
    ).fit(
        rating_matrix=rating_mat,
        implicit_matrix=implicit_mat,
        user_ids=mappings["user_ids"],
        item_ids=mappings["item_ids"],
        user_genre_profile=store.user_genre_profile,
        item_genre_vectors=store.item_genre_vectors,
        genre_columns=mappings["genre_columns"],
        train_df=train_df,
    )
    hyb_time = time.perf_counter() - t0

    hybrid_path = models_dir / "hybrid_svd_recommender.pkl"
    with hybrid_path.open("wb") as handle:
        pickle.dump(hybrid_model, handle)

    with mlflow.start_run(run_name="personalized_hybrid_svd") as hyb_run:
        mlflow.set_tags({"model_family": "personalized_hybrid", "personalized": "true", "feature_store_version": store.metadata["feature_store_version"]})
        mlflow.log_params(
            {
                "model_type": "HybridSVDRecommender",
                "n_factors": hyb_cfg["n_factors"],
                "n_epochs": hyb_cfg["n_epochs"],
                "reg": hyb_cfg["reg"],
                "bias_reg": hyb_cfg["bias_reg"],
                "implicit_svd_factors": hyb_cfg["implicit_svd_factors"],
                "explicit_weight": hyb_cfg["explicit_weight"],
                "implicit_weight": hyb_cfg["implicit_weight"],
                "genre_weight": hyb_cfg["genre_weight"],
                "train_rows": len(train_df),
                "n_users": mappings["n_users"],
                "n_items": mappings["n_items"],
            }
        )
        for epoch_idx, train_rmse in enumerate(hybrid_model.training_loss_history, start=1):
            mlflow.log_metric("train_rmse", train_rmse, step=epoch_idx)
        mlflow.log_metric("train_duration_sec", round(hyb_time, 4))
        mlflow.log_artifact(str(hybrid_path), artifact_path="model_artifacts")
        mlflow.log_artifact(str(feat_dir / "feature_metadata.json"), artifact_path="feature_store")
        hyb_run_id = hyb_run.info.run_id

    registry = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "mlflow_tracking_uri": cfg["training"]["mlflow_tracking_uri"],
        "mlflow_experiment_id": experiment.experiment_id,
        "mlflow_experiment_name": cfg["training"]["mlflow_experiment_name"],
        "models": {
            "baseline_popularity": {
                "run_id": pop_run_id,
                "artifact_path": str(pop_path.relative_to(PROJECT_ROOT)),
                "train_duration_sec": round(pop_time, 4),
                "params": pop_cfg,
            },
            "personalized_hybrid_svd": {
                "run_id": hyb_run_id,
                "artifact_path": str(hybrid_path.relative_to(PROJECT_ROOT)),
                "train_duration_sec": round(hyb_time, 4),
                "final_train_rmse": hybrid_model.training_loss_history[-1],
                "params": hyb_cfg,
            },
        },
    }
    registry_path = models_dir / "model_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    table = Table(title="Stage 6 Trained Models & MLflow Runs")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Type", style="dim", no_wrap=True)
    table.add_column("Train Time", justify="right", no_wrap=True)
    table.add_column("Train RMSE", justify="right", no_wrap=True)
    table.add_column("MLflow Run ID", style="green", no_wrap=True)

    table.add_row(
        "PopularityRecommender",
        "Baseline",
        f"{pop_time:.3f}s",
        "-",
        pop_run_id[:12] + "...",
    )
    table.add_row(
        "HybridSVDRecommender",
        "Personalized Hybrid",
        f"{hyb_time:.3f}s",
        f"{hybrid_model.training_loss_history[-1]:.4f}",
        hyb_run_id[:12] + "...",
    )
    console.print(table)
    logger.info("Models & MLflow runs logged; registry saved to %s", registry_path.relative_to(PROJECT_ROOT))
    return registry


if __name__ == "__main__":
    run_training()
