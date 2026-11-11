"""Stage 3: Data Cleaning, Standardization, and Time-Based Train/Test Split."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.table import Table

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


def _extract_release_year(row: pd.Series, default_year: int) -> int:
    """Extract release year from release_date or fallback to title suffix / default."""
    rel = row.get("release_date")
    if pd.notna(rel) and str(rel).strip():
        match = re.search(r"(\d{4})", str(rel))
        if match:
            return int(match.group(1))
    title = str(row.get("title", ""))
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        return int(match.group(1))
    return default_year


def _clean_users(users_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.drop_duplicates(subset=["user_id"], keep="first").copy()
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["occupation"] = (
        df["occupation"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
    df["zip_code"] = df["zip_code"].fillna("00000").astype(str).str.strip().str.zfill(5)
    df["zip_region"] = np.where(
        df["zip_code"].str[0].str.isdigit(),
        "region_" + df["zip_code"].str[0],
        "region_intl",
    )
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 19, 29, 39, 49, 120],
        labels=["<20", "20-29", "30-39", "40-49", "50+"],
    ).astype(str)
    return df.sort_values("user_id").reset_index(drop=True)


def _clean_items(items_df: pd.DataFrame, default_year: int) -> pd.DataFrame:
    df = items_df.drop_duplicates(subset=["item_id"], keep="first").copy()
    if "video_release_date" in df.columns:
        df = df.drop(columns=["video_release_date"])

    df["title"] = df["title"].astype(str).str.strip()
    df["imdb_url"] = df["imdb_url"].fillna("").astype(str).str.strip()
    df["release_year"] = df.apply(lambda r: _extract_release_year(r, default_year), axis=1).astype("int64")
    df["release_date"] = df["release_date"].fillna(f"01-Jan-{default_year}").astype(str)
    df["genres"] = df["genres"].fillna("Unknown").astype(str)
    df["primary_genre"] = df["genres"].str.split("|").str[0]

    genre_cols = [c for c in GENRE_COLUMNS if c in df.columns]
    df["genre_count"] = df[genre_cols].sum(axis=1).astype("int64")
    return df.sort_values("item_id").reset_index(drop=True)


def _clean_interactions(inter_df: pd.DataFrame) -> pd.DataFrame:
    df = inter_df.drop_duplicates(subset=["user_id", "item_id"], keep="first").copy()
    df["interaction_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["interaction_date"] = df["interaction_dt"].dt.strftime("%Y-%m-%d")
    df["interaction_hour"] = df["interaction_dt"].dt.hour.astype("int64")
    df["day_of_week"] = df["interaction_dt"].dt.dayofweek.astype("int64")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int64")
    return df.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)


def _temporal_split(
    inter_df: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform leakage-free per-user chronological train/test split."""
    ordered = inter_df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    ordered["_user_rank"] = ordered.groupby("user_id").cumcount()
    ordered["_user_total"] = ordered.groupby("user_id")["item_id"].transform("count")
    ordered["_train_cutoff"] = np.maximum(1, np.floor(ordered["_user_total"] * train_ratio).astype(int))

    is_train = ordered["_user_rank"] < ordered["_train_cutoff"]
    cols_to_drop = ["_user_rank", "_user_total", "_train_cutoff"]

    train_df = ordered.loc[is_train].drop(columns=cols_to_drop).reset_index(drop=True)
    test_df = ordered.loc[~is_train].drop(columns=cols_to_drop).reset_index(drop=True)
    return train_df, test_df


def run_preparation(config_path: str | Path | None = None) -> dict[str, Any]:
    """Clean validated data, standardize formats, and produce temporal train/test splits."""
    configure_runtime_env()
    cfg = load_config(config_path)
    paths = ensure_directories(cfg)
    logger = get_logger("stage.preparation")
    console = get_console()

    print_stage_banner(
        "Stage 3",
        "Data Preparation & Temporal Split",
        f"Strategy: {cfg['preparation']['split_strategy']} ({round(cfg['preparation']['train_ratio'] * 100)}/"
        f"{round((1 - cfg['preparation']['train_ratio']) * 100)})",
    )

    proc_dir = paths["processed_dir"]
    users_val = pd.read_parquet(proc_dir / "validated_users.parquet")
    items_val = pd.read_parquet(proc_dir / "validated_items.parquet")
    inter_val = pd.read_parquet(proc_dir / "validated_interactions.parquet")

    users_clean = _clean_users(users_val)
    items_clean = _clean_items(items_val, int(cfg["preparation"]["default_release_year"]))
    inter_clean = _clean_interactions(inter_val)

    train_ratio = float(cfg["preparation"]["train_ratio"])
    train_df, test_df = _temporal_split(inter_clean, train_ratio=train_ratio)

    users_clean.to_parquet(proc_dir / "users_clean.parquet", index=False)
    items_clean.to_parquet(proc_dir / "items_clean.parquet", index=False)
    inter_clean.to_parquet(proc_dir / "interactions_clean.parquet", index=False)
    train_df.to_parquet(proc_dir / "train_interactions.parquet", index=False)
    test_df.to_parquet(proc_dir / "test_interactions.parquet", index=False)

    summary = {
        "split_strategy": cfg["preparation"]["split_strategy"],
        "train_ratio": train_ratio,
        "users_clean_count": len(users_clean),
        "items_clean_count": len(items_clean),
        "interactions_clean_count": len(inter_clean),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_users": int(train_df["user_id"].nunique()),
        "test_users": int(test_df["user_id"].nunique()),
        "user_overlap": int(len(set(train_df["user_id"]) & set(test_df["user_id"]))),
        "train_items": int(train_df["item_id"].nunique()),
        "test_items": int(test_df["item_id"].nunique()),
        "min_timestamp_utc": str(inter_clean["interaction_dt"].min()),
        "max_timestamp_utc": str(inter_clean["interaction_dt"].max()),
        "dropped_columns": ["video_release_date (100% null in source)"],
    }
    summary_path = proc_dir / "preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table = Table(title="Stage 3 Preparation & Temporal Split Summary")
    table.add_column("Artifact", style="cyan", no_wrap=True)
    table.add_column("Rows", justify="right", no_wrap=True, style="bold green")
    table.add_column("Unique Users", justify="right", no_wrap=True)
    table.add_column("Unique Items", justify="right", no_wrap=True)
    table.add_column("Notes", style="dim")

    table.add_row("users_clean.parquet", f"{len(users_clean):,}", f"{len(users_clean):,}", "-", "Normalized gender, occupation, zip_region, age_group")
    table.add_row("items_clean.parquet", f"{len(items_clean):,}", "-", f"{len(items_clean):,}", "Dropped 100%-null video_release_date; imputed release_year")
    table.add_row("train_interactions.parquet", f"{len(train_df):,}", f"{summary['train_users']:,}", f"{summary['train_items']:,}", "Earliest 80% interactions per user")
    table.add_row("test_interactions.parquet", f"{len(test_df):,}", f"{summary['test_users']:,}", f"{summary['test_items']:,}", "Most recent 20% interactions per user (held out)")
    console.print(table)
    logger.info(
        "Temporal split complete: %s train / %s test (%d overlapping users). Saved to %s",
        f"{len(train_df):,}",
        f"{len(test_df):,}",
        summary["user_overlap"],
        summary_path.relative_to(PROJECT_ROOT),
    )
    return summary


if __name__ == "__main__":
    run_preparation()
