"""Stage 5: Reusable Feature Store & Feature Catalog Generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
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

FEATURE_CATALOG_SPECS: list[dict[str, str]] = [
    {"feature": "user_avg_rating", "entity": "user", "dtype": "float64", "definition": "Mean historical rating given by user on train split", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "user_rating_std", "entity": "user", "dtype": "float64", "definition": "Standard deviation of user's training ratings", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "user_activity_count", "entity": "user", "dtype": "int64", "definition": "Total number of interactions in user's train history", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "user_positive_rate", "entity": "user", "dtype": "float64", "definition": "Share of user ratings >= 4.0", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "user_preferred_genre", "entity": "user", "dtype": "string", "definition": "Top genre weighted by user ratings across rated items", "source": "train_interactions + items", "reuse": "Training & Inference"},
    {"feature": "user_genre_entropy", "entity": "user", "dtype": "float64", "definition": "Shannon entropy over user's rated genre distribution", "source": "train_interactions + items", "reuse": "Training & Inference"},
    {"feature": "user_days_since_last_interaction", "entity": "user", "dtype": "float64", "definition": "Days from user's last train interaction to train cutoff", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "user_age", "entity": "user", "dtype": "int64", "definition": "Validated user age in years", "source": "users_clean", "reuse": "Training & Inference"},
    {"feature": "user_gender_code", "entity": "user", "dtype": "int64", "definition": "Binary encoding of gender (M=1, F=0)", "source": "users_clean", "reuse": "Training & Inference"},
    {"feature": "user_occupation_code", "entity": "user", "dtype": "int64", "definition": "Deterministic integer encoding of occupation category", "source": "users_clean", "reuse": "Training & Inference"},
    {"feature": "item_popularity_count", "entity": "item", "dtype": "int64", "definition": "Total training rating count received by item", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "item_popularity_log", "entity": "item", "dtype": "float64", "definition": "Log-scaled popularity log(1 + item_popularity_count)", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "item_avg_rating", "entity": "item", "dtype": "float64", "definition": "Mean training rating received by item (prior fallback)", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "item_bayesian_rating", "entity": "item", "dtype": "float64", "definition": "Bayesian smoothed score (v*R + m*C)/(v+m)", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "item_positive_rate", "entity": "item", "dtype": "float64", "definition": "Fraction of item ratings >= 4.0", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "item_release_year", "entity": "item", "dtype": "int64", "definition": "Cleaned movie release year", "source": "items_clean", "reuse": "Training & Inference"},
    {"feature": "item_age_years", "entity": "item", "dtype": "int64", "definition": "Movie age in years relative to dataset reference year (1998)", "source": "items_clean", "reuse": "Training & Inference"},
    {"feature": "item_primary_genre", "entity": "item", "dtype": "string", "definition": "Primary genre label assigned to item", "source": "items_clean", "reuse": "Training & Inference"},
    {"feature": "item_genre_count", "entity": "item", "dtype": "int64", "definition": "Number of genre tags attached to movie", "source": "items_clean", "reuse": "Training & Inference"},
    {"feature": "item_days_since_last_interaction", "entity": "item", "dtype": "float64", "definition": "Days elapsed since item's most recent train rating", "source": "train_interactions", "reuse": "Training & Inference"},
    {"feature": "user_item_bias_baseline", "entity": "interaction", "dtype": "float64", "definition": "Additive baseline global_mean + b_u + b_i clipped to [1, 5]", "source": "user + item features", "reuse": "Training & Inference"},
    {"feature": "user_preferred_genre_match", "entity": "interaction", "dtype": "float64", "definition": "Cosine affinity between user genre profile and item genres", "source": "user_genre_profile + items", "reuse": "Training & Inference"},
    {"feature": "is_preferred_genre", "entity": "interaction", "dtype": "int64", "definition": "Indicator (1/0) whether item includes user's top genre", "source": "user + item features", "reuse": "Training & Inference"},
    {"feature": "interaction_recency_days", "entity": "interaction", "dtype": "float64", "definition": "Days elapsed between interaction timestamp and train cutoff", "source": "train_interactions", "reuse": "Training Only"},
    {"feature": "is_weekend_interaction", "entity": "interaction", "dtype": "int64", "definition": "Indicator (1/0) whether interaction occurred on Sat/Sun", "source": "train_interactions", "reuse": "Training & Context"},
    {"feature": "rating_deviation_from_user_mean", "entity": "interaction", "dtype": "float64", "definition": "Observed rating minus user_avg_rating", "source": "train_interactions", "reuse": "Training Target Residual"},
]


@dataclass
class FeatureStore:
    """Offline & online feature retrieval layer backed by materialized Parquet feature tables."""

    user_features: pd.DataFrame
    item_features: pd.DataFrame
    user_genre_profile: pd.DataFrame
    item_genre_vectors: pd.DataFrame
    metadata: dict[str, Any]
    features_dir: Path

    @classmethod
    def load(cls, features_dir: str | Path) -> "FeatureStore":
        fdir = Path(features_dir)
        return cls(
            user_features=pd.read_parquet(fdir / "user_features.parquet"),
            item_features=pd.read_parquet(fdir / "item_features.parquet"),
            user_genre_profile=pd.read_parquet(fdir / "user_genre_profile.parquet"),
            item_genre_vectors=pd.read_parquet(fdir / "item_genre_vectors.parquet"),
            metadata=json.loads((fdir / "feature_metadata.json").read_text(encoding="utf-8")),
            features_dir=fdir,
        )

    def get_training_features(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load materialized user, item, and interaction feature tables for training."""
        inter_feats = pd.read_parquet(self.features_dir / "interaction_features.parquet")
        return self.user_features, self.item_features, inter_feats

    def build_candidate_features(self, user_id: int, candidate_item_ids: list[int]) -> pd.DataFrame:
        """Assemble real-time inference features for a (user, candidate_items) scoring batch."""
        global_mean = float(self.metadata["global_mean_rating"])
        u_row = self.user_features[self.user_features["user_id"] == user_id]
        if u_row.empty:
            user_avg = global_mean
            user_act = 0
            user_pref = "Unknown"
            user_recency = float(self.metadata["max_train_span_days"])
            u_vec = np.zeros(len(GENRE_COLUMNS), dtype=np.float64)
        else:
            r = u_row.iloc[0]
            user_avg = float(r["user_avg_rating"])
            user_act = int(r["user_activity_count"])
            user_pref = str(r["user_preferred_genre"])
            user_recency = float(r["user_days_since_last_interaction"])
            ug_row = self.user_genre_profile[self.user_genre_profile["user_id"] == user_id]
            u_vec = ug_row[GENRE_COLUMNS].to_numpy(dtype=np.float64)[0] if not ug_row.empty else np.zeros(len(GENRE_COLUMNS))

        cand = self.item_features[self.item_features["item_id"].isin(candidate_item_ids)].copy()
        item_order = {iid: pos for pos, iid in enumerate(candidate_item_ids)}
        cand["_order"] = cand["item_id"].map(item_order)
        cand = cand.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

        ig_aligned = self.item_genre_vectors.set_index("item_id").loc[cand["item_id"], GENRE_COLUMNS].to_numpy(dtype=np.float64)
        u_norm = np.linalg.norm(u_vec) + 1e-12
        i_norms = np.linalg.norm(ig_aligned, axis=1) + 1e-12
        genre_sim = (ig_aligned @ u_vec) / (u_norm * i_norms)

        cand["user_id"] = user_id
        cand["user_avg_rating"] = user_avg
        cand["user_activity_count"] = user_act
        cand["user_preferred_genre"] = user_pref
        cand["user_days_since_last_interaction"] = user_recency
        cand["user_item_bias_baseline"] = np.clip(
            global_mean + (user_avg - global_mean) + (cand["item_avg_rating"] - global_mean),
            1.0,
            5.0,
        )
        cand["user_preferred_genre_match"] = np.round(genre_sim, 4)
        pref_col = f"genre_{user_pref.lower().replace(' ', '_').replace('-', '_')}"
        if pref_col in GENRE_COLUMNS:
            col_idx = GENRE_COLUMNS.index(pref_col)
            cand["is_preferred_genre"] = (ig_aligned[:, col_idx] > 0).astype("int64")
        else:
            cand["is_preferred_genre"] = (cand["item_primary_genre"] == user_pref).astype("int64")

        cols = [
            "user_id",
            "item_id",
            "user_avg_rating",
            "user_activity_count",
            "user_preferred_genre",
            "item_popularity_count",
            "item_avg_rating",
            "item_bayesian_rating",
            "item_primary_genre",
            "user_item_bias_baseline",
            "user_preferred_genre_match",
            "is_preferred_genre",
        ]
        return cand[cols]


def _compute_genre_label(col_name: str) -> str:
    label = col_name.replace("genre_", "")
    if label == "sci_fi":
        return "Sci Fi"
    if label == "film_noir":
        return "Film Noir"
    return label.replace("_", " ").title()


def _write_feature_catalog(reports_dir: Path) -> None:
    catalog_df = pd.DataFrame(FEATURE_CATALOG_SPECS)
    catalog_df.to_csv(reports_dir / "feature_catalog.csv", index=False)

    lines = [
        "# Feature Catalog",
        "",
        f"Total registered features: **{len(FEATURE_CATALOG_SPECS)}** across `user` (10), `item` (10), and `interaction` (6) entities.",
        "",
        "| # | Feature Name | Entity | Data Type | Computation / Definition | Source | Reuse Scope |",
        "|---:|---|---|---|---|---|---|",
    ]
    for idx, spec in enumerate(FEATURE_CATALOG_SPECS, start=1):
        lines.append(
            f"| {idx} | `{spec['feature']}` | `{spec['entity']}` | `{spec['dtype']}` | "
            f"{spec['definition']} | `{spec['source']}` | {spec['reuse']} |"
        )
    (reports_dir / "feature_catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_engineering(config_path: str | Path | None = None) -> dict[str, Any]:
    """Compute, materialize, and catalog reusable user, item, and interaction features."""
    configure_runtime_env()
    cfg = load_config(config_path)
    paths = ensure_directories(cfg)
    logger = get_logger("stage.features")
    console = get_console()

    print_stage_banner(
        "Stage 5",
        "Feature Management & Feature Store",
        f"Output Store: {cfg['paths']['features_dir']} | Catalog: {cfg['paths']['reports_dir']}/feature_catalog.md",
    )

    proc_dir = paths["processed_dir"]
    feat_dir = paths["features_dir"]
    rep_dir = paths["reports_dir"]

    users_df = pd.read_parquet(proc_dir / "users_clean.parquet")
    items_df = pd.read_parquet(proc_dir / "items_clean.parquet")
    train_df = pd.read_parquet(proc_dir / "train_interactions.parquet")
    user_summary = pd.read_parquet(proc_dir / "user_activity_summary.parquet")
    item_pop = pd.read_parquet(proc_dir / "item_popularity.parquet")

    ref_ts = int(train_df["timestamp"].max())
    min_ts = int(train_df["timestamp"].min())
    global_mean = float(train_df["rating"].mean())
    max_span_days = (ref_ts - min_ts) / 86400.0
    ref_year = pd.to_datetime(ref_ts, unit="s", utc=True).year

    # User genre affinity matrix (weighted by user rating)
    merged_genres = train_df[["user_id", "item_id", "rating"]].merge(
        items_df[["item_id", *GENRE_COLUMNS]], on="item_id", how="left"
    )
    weighted_genres = merged_genres[GENRE_COLUMNS].multiply(merged_genres["rating"], axis=0)
    weighted_genres["user_id"] = merged_genres["user_id"]
    user_genre_sum = weighted_genres.groupby("user_id")[GENRE_COLUMNS].sum()
    user_genre_sum = user_genre_sum.reindex(users_df["user_id"].tolist(), fill_value=0.0)

    row_sums = user_genre_sum.sum(axis=1).replace(0.0, 1.0)
    user_genre_prob = user_genre_sum.div(row_sums, axis=0)

    # Shannon entropy over genre distribution
    probs_arr = user_genre_prob.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(probs_arr > 0, np.log2(probs_arr), 0.0)
    genre_entropy = -np.sum(probs_arr * log_p, axis=1)

    top_genre_col = user_genre_sum.idxmax(axis=1)
    preferred_genre = top_genre_col.map(_compute_genre_label)

    occ_categories = sorted(users_df["occupation"].unique().tolist())
    occ_to_code = {occ: idx for idx, occ in enumerate(occ_categories)}

    user_features = user_summary[
        ["user_id", "mean_rating", "std_rating", "total_ratings", "positive_rate", "last_interaction_ts", "age", "gender", "occupation"]
    ].rename(
        columns={
            "mean_rating": "user_avg_rating",
            "std_rating": "user_rating_std",
            "total_ratings": "user_activity_count",
            "positive_rate": "user_positive_rate",
            "age": "user_age",
        }
    )
    user_features["user_preferred_genre"] = user_features["user_id"].map(preferred_genre)
    user_features["user_genre_entropy"] = pd.Series(genre_entropy, index=user_genre_sum.index).reindex(user_features["user_id"]).to_numpy()
    user_features["user_days_since_last_interaction"] = (ref_ts - user_features["last_interaction_ts"]) / 86400.0
    user_features["user_gender_code"] = (user_features["gender"] == "M").astype("int64")
    user_features["user_occupation_code"] = user_features["occupation"].map(occ_to_code).astype("int64")
    user_features = user_features[
        [
            "user_id",
            "user_avg_rating",
            "user_rating_std",
            "user_activity_count",
            "user_positive_rate",
            "user_preferred_genre",
            "user_genre_entropy",
            "user_days_since_last_interaction",
            "user_age",
            "user_gender_code",
            "user_occupation_code",
        ]
    ].sort_values("user_id").reset_index(drop=True)

    user_genre_profile = user_genre_prob.reset_index().rename(columns={"index": "user_id"})
    user_genre_profile.columns = ["user_id", *GENRE_COLUMNS]

    item_features = item_pop[
        [
            "item_id",
            "rating_count",
            "mean_rating",
            "bayesian_rating",
            "positive_rate",
            "release_year",
            "primary_genre",
            "genre_count",
            "last_interaction_ts",
        ]
    ].rename(
        columns={
            "rating_count": "item_popularity_count",
            "mean_rating": "item_avg_rating",
            "bayesian_rating": "item_bayesian_rating",
            "positive_rate": "item_positive_rate",
            "release_year": "item_release_year",
            "primary_genre": "item_primary_genre",
            "genre_count": "item_genre_count",
        }
    )
    item_features["item_popularity_log"] = np.log1p(item_features["item_popularity_count"])
    item_features["item_age_years"] = np.maximum(0, ref_year - item_features["item_release_year"]).astype("int64")
    item_features["item_days_since_last_interaction"] = np.where(
        item_features["last_interaction_ts"].notna(),
        (ref_ts - item_features["last_interaction_ts"].fillna(min_ts)) / 86400.0,
        max_span_days,
    )
    item_features = item_features[
        [
            "item_id",
            "item_popularity_count",
            "item_popularity_log",
            "item_avg_rating",
            "item_bayesian_rating",
            "item_positive_rate",
            "item_release_year",
            "item_age_years",
            "item_primary_genre",
            "item_genre_count",
            "item_days_since_last_interaction",
        ]
    ].sort_values("item_id").reset_index(drop=True)

    item_genre_vectors = items_df[["item_id", *GENRE_COLUMNS]].sort_values("item_id").reset_index(drop=True)

    # Interaction features for training interactions
    inter = train_df[["user_id", "item_id", "rating", "timestamp", "is_weekend"]].merge(
        user_features[["user_id", "user_avg_rating", "user_preferred_genre"]], on="user_id", how="left"
    ).merge(
        item_features[["item_id", "item_avg_rating", "item_primary_genre"]], on="item_id", how="left"
    )

    ug_lookup = user_genre_profile.set_index("user_id")[GENRE_COLUMNS]
    ig_lookup = item_genre_vectors.set_index("item_id")[GENRE_COLUMNS]
    u_mat = ug_lookup.loc[inter["user_id"]].to_numpy(dtype=np.float64)
    i_mat = ig_lookup.loc[inter["item_id"]].to_numpy(dtype=np.float64)
    dot = np.sum(u_mat * i_mat, axis=1)
    norms = (np.linalg.norm(u_mat, axis=1) + 1e-12) * (np.linalg.norm(i_mat, axis=1) + 1e-12)

    inter["user_item_bias_baseline"] = np.clip(
        global_mean + (inter["user_avg_rating"] - global_mean) + (inter["item_avg_rating"] - global_mean),
        1.0,
        5.0,
    )
    inter["user_preferred_genre_match"] = np.round(dot / norms, 4)

    pref_col_series = "genre_" + inter["user_preferred_genre"].str.lower().str.replace(" ", "_").str.replace("-", "_")
    col_to_idx = {col: idx for idx, col in enumerate(GENRE_COLUMNS)}
    col_indices = pref_col_series.map(col_to_idx).fillna(-1).astype(int).to_numpy()
    matched_flag = np.where(
        col_indices >= 0,
        i_mat[np.arange(len(inter)), np.maximum(col_indices, 0)] > 0,
        inter["item_primary_genre"] == inter["user_preferred_genre"],
    )
    inter["is_preferred_genre"] = matched_flag.astype("int64")
    inter["interaction_recency_days"] = (ref_ts - inter["timestamp"]) / 86400.0
    inter["is_weekend_interaction"] = inter["is_weekend"].astype("int64")
    inter["rating_deviation_from_user_mean"] = inter["rating"] - inter["user_avg_rating"]

    interaction_features = inter[
        [
            "user_id",
            "item_id",
            "rating",
            "timestamp",
            "user_item_bias_baseline",
            "user_preferred_genre_match",
            "is_preferred_genre",
            "interaction_recency_days",
            "is_weekend_interaction",
            "rating_deviation_from_user_mean",
        ]
    ].reset_index(drop=True)

    user_features.to_parquet(feat_dir / "user_features.parquet", index=False)
    item_features.to_parquet(feat_dir / "item_features.parquet", index=False)
    interaction_features.to_parquet(feat_dir / "interaction_features.parquet", index=False)
    user_genre_profile.to_parquet(feat_dir / "user_genre_profile.parquet", index=False)
    item_genre_vectors.to_parquet(feat_dir / "item_genre_vectors.parquet", index=False)

    metadata = {
        "feature_store_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ref_timestamp": ref_ts,
        "ref_year": int(ref_year),
        "global_mean_rating": round(global_mean, 6),
        "max_train_span_days": round(max_span_days, 4),
        "occupation_vocab": occ_to_code,
        "genre_columns": GENRE_COLUMNS,
        "feature_counts": {
            "user_features": len(user_features.columns) - 1,
            "item_features": len(item_features.columns) - 1,
            "interaction_features": 6,
            "total_catalog_features": len(FEATURE_CATALOG_SPECS),
        },
        "row_counts": {
            "user_features": len(user_features),
            "item_features": len(item_features),
            "interaction_features": len(interaction_features),
        },
    }
    (feat_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_feature_catalog(rep_dir)

    # Verify serving / online reuse path
    store = FeatureStore.load(feat_dir)
    sample_uid = int(cfg["evaluation"]["sample_user_id"])
    sample_candidates = [1, 50, 100, 181, 258]
    served_df = store.build_candidate_features(sample_uid, sample_candidates)

    table = Table(title="Stage 5 Feature Store Materialization")
    table.add_column("Feature Table", style="cyan", no_wrap=True)
    table.add_column("Rows", justify="right", no_wrap=True, style="bold green")
    table.add_column("Cols", justify="right", no_wrap=True)
    table.add_column("Artifact File", style="dim")

    table.add_row("user_features", f"{len(user_features):,}", str(len(user_features.columns) - 1), "user_features.parquet")
    table.add_row("item_features", f"{len(item_features):,}", str(len(item_features.columns) - 1), "item_features.parquet")
    table.add_row("interaction_features", f"{len(interaction_features):,}", "6", "interaction_features.parquet")
    console.print(table)

    serve_table = Table(title=f"Online Inference Feature Serving Check (user_id={sample_uid})")
    serve_table.add_column("Item", justify="right", style="cyan", no_wrap=True)
    serve_table.add_column("UserPref", style="dim", no_wrap=True)
    serve_table.add_column("ItemGenre", style="dim", no_wrap=True)
    serve_table.add_column("Bayes", justify="right", no_wrap=True)
    serve_table.add_column("BiasPred", justify="right", no_wrap=True)
    serve_table.add_column("GenreSim", justify="right", no_wrap=True, style="bold green")
    serve_table.add_column("PrefHit", justify="right", no_wrap=True)

    for _, r in served_df.iterrows():
        serve_table.add_row(
            str(int(r["item_id"])),
            str(r["user_preferred_genre"]),
            str(r["item_primary_genre"]),
            f"{r['item_bayesian_rating']:.3f}",
            f"{r['user_item_bias_baseline']:.3f}",
            f"{r['user_preferred_genre_match']:.4f}",
            str(int(r["is_preferred_genre"])),
        )
    console.print(serve_table)
    logger.info("Feature catalog exported to reports/feature_catalog.md (%d features)", len(FEATURE_CATALOG_SPECS))
    return metadata


if __name__ == "__main__":
    run_feature_engineering()
