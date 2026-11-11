"""Stage 4: Derived Datasets & Sparse User-Item Interaction Matrix Construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.table import Table
from scipy import sparse

from src.ingestion.ingest import GENRE_COLUMNS
from src.utils.config import (
    PROJECT_ROOT,
    configure_runtime_env,
    ensure_directories,
    get_console,
    get_logger,
    load_config,
    print_stage_banner,
)


def _build_user_activity_summary(
    users_df: pd.DataFrame,
    train_df: pd.DataFrame,
    items_df: pd.DataFrame,
    pos_thresh: float,
) -> pd.DataFrame:
    genre_cols = [c for c in GENRE_COLUMNS if c in items_df.columns]
    merged = train_df.merge(items_df[["item_id", *genre_cols]], on="item_id", how="left")

    agg = train_df.groupby("user_id").agg(
        total_ratings=("rating", "count"),
        positive_ratings=("rating", lambda s: int((s >= pos_thresh).sum())),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        min_rating=("rating", "min"),
        max_rating=("rating", "max"),
        first_interaction_ts=("timestamp", "min"),
        last_interaction_ts=("timestamp", "max"),
    ).reset_index()

    agg["std_rating"] = agg["std_rating"].fillna(0.0)
    agg["positive_rate"] = agg["positive_ratings"] / agg["total_ratings"]
    agg["active_span_days"] = (agg["last_interaction_ts"] - agg["first_interaction_ts"]) / 86400.0

    genre_coverage = (merged.groupby("user_id")[genre_cols].sum() > 0).sum(axis=1).rename("distinct_genres_rated")
    summary = users_df[["user_id", "age", "gender", "occupation", "age_group"]].merge(agg, on="user_id", how="left")
    summary = summary.merge(genre_coverage, on="user_id", how="left")
    return summary.sort_values("user_id").reset_index(drop=True)


def _build_item_popularity(
    items_df: pd.DataFrame,
    train_df: pd.DataFrame,
    pos_thresh: float,
    bayesian_m: float,
) -> pd.DataFrame:
    global_mean = float(train_df["rating"].mean())
    agg = train_df.groupby("item_id").agg(
        rating_count=("rating", "count"),
        positive_count=("rating", lambda s: int((s >= pos_thresh).sum())),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        last_interaction_ts=("timestamp", "max"),
    ).reset_index()

    pop = items_df[["item_id", "title", "release_year", "primary_genre", "genres", "genre_count"]].merge(
        agg, on="item_id", how="left"
    )
    pop["rating_count"] = pop["rating_count"].fillna(0).astype("int64")
    pop["positive_count"] = pop["positive_count"].fillna(0).astype("int64")
    pop["mean_rating"] = pop["mean_rating"].fillna(global_mean)
    pop["std_rating"] = pop["std_rating"].fillna(0.0)
    pop["positive_rate"] = np.where(pop["rating_count"] > 0, pop["positive_count"] / pop["rating_count"], 0.0)
    pop["bayesian_rating"] = (
        (pop["rating_count"] * pop["mean_rating"] + bayesian_m * global_mean)
        / (pop["rating_count"] + bayesian_m)
    )
    pop["popularity_score"] = pop["bayesian_rating"] + 0.15 * np.log1p(pop["rating_count"])
    pop = pop.sort_values(["popularity_score", "rating_count", "item_id"], ascending=[False, False, True]).reset_index(drop=True)
    pop["popularity_rank"] = np.arange(1, len(pop) + 1)
    return pop.sort_values("item_id").reset_index(drop=True)


def _build_interaction_matrices(
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    train_df: pd.DataFrame,
    pos_thresh: float,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
    user_ids = sorted(users_df["user_id"].astype(int).tolist())
    item_ids = sorted(items_df["item_id"].astype(int).tolist())
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    rows = train_df["user_id"].map(user_to_idx).to_numpy()
    cols = train_df["item_id"].map(item_to_idx).to_numpy()
    ratings = train_df["rating"].to_numpy(dtype=np.float32)
    implicit_vals = (ratings >= pos_thresh).astype(np.float32)

    n_users, n_items = len(user_ids), len(item_ids)
    rating_mat = sparse.csr_matrix((ratings, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    implicit_mat = sparse.csr_matrix((implicit_vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

    genre_cols = [c for c in GENRE_COLUMNS if c in items_df.columns]
    items_aligned = items_df.set_index("item_id").loc[item_ids]
    genre_mat = sparse.csr_matrix(items_aligned[genre_cols].to_numpy(dtype=np.float32))

    mappings = {
        "n_users": n_users,
        "n_items": n_items,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "genre_columns": genre_cols,
    }
    return rating_mat, implicit_mat, genre_mat, mappings


def run_transformation(config_path: str | Path | None = None) -> dict[str, Any]:
    """Create derived user/item summaries and sparse CSR interaction matrices from train split."""
    configure_runtime_env()
    cfg = load_config(config_path)
    paths = ensure_directories(cfg)
    logger = get_logger("stage.transformation")
    console = get_console()

    print_stage_banner(
        "Stage 4",
        "Data Transformation & Derived Datasets",
        f"Source: {cfg['paths']['processed_dir']} (leakage-free train split)",
    )

    proc_dir = paths["processed_dir"]
    users_df = pd.read_parquet(proc_dir / "users_clean.parquet")
    items_df = pd.read_parquet(proc_dir / "items_clean.parquet")
    train_df = pd.read_parquet(proc_dir / "train_interactions.parquet")

    pos_thresh = float(cfg["features"]["positive_rating_threshold"])
    bayesian_m = float(cfg["features"]["bayesian_smoothing_m"])

    user_summary = _build_user_activity_summary(users_df, train_df, items_df, pos_thresh=pos_thresh)
    item_popularity = _build_item_popularity(items_df, train_df, pos_thresh=pos_thresh, bayesian_m=bayesian_m)
    rating_mat, implicit_mat, genre_mat, mappings = _build_interaction_matrices(
        users_df, items_df, train_df, pos_thresh=pos_thresh
    )

    user_summary.to_parquet(proc_dir / "user_activity_summary.parquet", index=False)
    item_popularity.to_parquet(proc_dir / "item_popularity.parquet", index=False)
    sparse.save_npz(proc_dir / "user_item_rating_matrix.npz", rating_mat)
    sparse.save_npz(proc_dir / "user_item_implicit_matrix.npz", implicit_mat)
    sparse.save_npz(proc_dir / "item_genre_matrix.npz", genre_mat)
    (proc_dir / "id_mappings.json").write_text(json.dumps(mappings), encoding="utf-8")

    density_pct = (rating_mat.nnz / (rating_mat.shape[0] * rating_mat.shape[1])) * 100.0
    summary = {
        "user_summary_rows": len(user_summary),
        "item_popularity_rows": len(item_popularity),
        "matrix_shape": list(rating_mat.shape),
        "matrix_nnz": int(rating_mat.nnz),
        "implicit_positive_nnz": int(implicit_mat.sum()),
        "matrix_density_pct": round(density_pct, 3),
        "matrix_sparsity_pct": round(100.0 - density_pct, 3),
    }
    summary_path = proc_dir / "transformation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table = Table(title="Stage 4 Derived Datasets & Interaction Matrices")
    table.add_column("Derived Artifact", style="cyan", no_wrap=True)
    table.add_column("Shape / Rows", justify="right", no_wrap=True, style="bold green")
    table.add_column("Details", style="dim")

    table.add_row("user_activity_summary.parquet", f"{len(user_summary):,} rows", "Avg rating, activity count, span days, genre breadth")
    table.add_row("item_popularity.parquet", f"{len(item_popularity):,} rows", "Popularity rank, Bayesian weighted rating, positive rate")
    table.add_row("user_item_rating_matrix.npz", f"{rating_mat.shape[0]} x {rating_mat.shape[1]}", f"CSR float32 | NNZ={rating_mat.nnz:,} | Sparsity={summary['matrix_sparsity_pct']:.2f}%")
    table.add_row("user_item_implicit_matrix.npz", f"{implicit_mat.shape[0]} x {implicit_mat.shape[1]}", f"Positive ratings >= {pos_thresh} | Positives={summary['implicit_positive_nnz']:,}")
    table.add_row("item_genre_matrix.npz", f"{genre_mat.shape[0]} x {genre_mat.shape[1]}", "Multi-hot item genre matrix aligned to item_idx")
    console.print(table)
    logger.info("Derived datasets saved to %s", proc_dir.relative_to(PROJECT_ROOT))
    return summary


if __name__ == "__main__":
    run_transformation()
