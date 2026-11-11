"""Stage 7: Model Evaluation (Ranking, Coverage, Diversity, RMSE/MAE) & MLflow Logging."""

from __future__ import annotations

import json
import pickle
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from rich.table import Table

from src.ingestion.ingest import GENRE_COLUMNS
from src.training.train import _resolve_mlflow_uri


def _compute_ndcg(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for rank, iid in enumerate(recommended[:k], start=1):
        if iid in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def _compute_intra_list_diversity(
    recommended: list[int],
    item_genre_norm_map: dict[int, np.ndarray],
) -> float:
    if len(recommended) < 2:
        return 0.0
    vecs = np.vstack([item_genre_norm_map[iid] for iid in recommended])
    sim_mat = vecs @ vecs.T
    n = len(recommended)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_mat[upper_indices]
    return float(np.mean(1.0 - np.clip(pairwise_sims, 0.0, 1.0)))


def _evaluate_model(
    model_name: str,
    model: Any,
    test_df: pd.DataFrame,
    all_item_ids: list[int],
    item_genre_norm_map: dict[int, np.ndarray],
    k_values: list[int],
    primary_k: int,
    relevance_threshold: float,
) -> tuple[dict[str, Any], dict[int, list[tuple[int, float]]]]:
    y_true = test_df["rating"].to_numpy(dtype=np.float64)
    y_pred = model.predict_ratings(
        test_df["user_id"].to_numpy(dtype=np.int64),
        test_df["item_id"].to_numpy(dtype=np.int64),
    )
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    relevant_by_user: dict[int, set[int]] = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["item_id"]
        .apply(lambda s: set(int(x) for x in s))
        .to_dict()
    )

    max_k = max(k_values)
    all_test_users = sorted(test_df["user_id"].unique().tolist())
    recs_by_user: dict[int, list[tuple[int, float]]] = {}
    for uid in all_test_users:
        recs_by_user[int(uid)] = model.recommend_for_user(int(uid), top_k=max_k, exclude_seen=True)

    metrics: dict[str, Any] = {
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "evaluated_users_for_ranking": len(relevant_by_user),
    }

    for k in sorted(k_values):
        precisions, recalls, ndcgs, diversities = [], [], [], []
        recommended_catalog: set[int] = set()

        for uid in all_test_users:
            rec_items = [iid for iid, _ in recs_by_user[int(uid)][:k]]
            recommended_catalog.update(rec_items)
            diversities.append(_compute_intra_list_diversity(rec_items, item_genre_norm_map))

            rel_set = relevant_by_user.get(int(uid))
            if rel_set:
                hits = len(set(rec_items) & rel_set)
                precisions.append(hits / k)
                recalls.append(hits / len(rel_set))
                ndcgs.append(_compute_ndcg(rec_items, rel_set, k))

        coverage = len(recommended_catalog) / len(all_item_ids)
        metrics[f"precision_at_{k}"] = round(float(np.mean(precisions)), 4)
        metrics[f"recall_at_{k}"] = round(float(np.mean(recalls)), 4)
        metrics[f"ndcg_at_{k}"] = round(float(np.mean(ndcgs)), 4)
        metrics[f"coverage_at_{k}"] = round(float(coverage), 4)
        metrics[f"diversity_at_{k}"] = round(float(np.mean(diversities)), 4)

    metrics["primary_k"] = primary_k
    return metrics, recs_by_user


def _save_comparison_chart(results_df: pd.DataFrame, primary_k: int, out_path: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cbd5e1"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [2.2, 1.0]})

    ranking_cols = [
        f"precision_at_{primary_k}",
        f"recall_at_{primary_k}",
        f"ndcg_at_{primary_k}",
        f"coverage_at_{primary_k}",
        f"diversity_at_{primary_k}",
    ]
    ranking_labels = [
        f"Precision@{primary_k}",
        f"Recall@{primary_k}",
        f"NDCG@{primary_k}",
        f"Coverage@{primary_k}",
        f"Diversity@{primary_k}",
    ]

    x = np.arange(len(ranking_labels))
    width = 0.34
    base_vals = [float(results_df.iloc[0][c]) for c in ranking_cols]
    hyb_vals = [float(results_df.iloc[1][c]) for c in ranking_cols]

    bars1 = ax1.bar(x - width / 2, base_vals, width, label=results_df.iloc[0]["model"], color="#64748b", edgecolor="#334155")
    bars2 = ax1.bar(x + width / 2, hyb_vals, width, label=results_df.iloc[1]["model"], color="#2563eb", edgecolor="#1e3a8a")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ranking_labels, fontweight="bold")
    ax1.set_ylim(0, max(max(base_vals), max(hyb_vals)) * 1.18)
    ax1.set_title(f"Top-{primary_k} Ranking, Coverage & Diversity (Higher is Better)", fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, facecolor="white")
    for b in [*bars1, *bars2]:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=8.5, xytext=(0, 3), textcoords="offset points")

    err_labels = ["RMSE", "MAE"]
    x2 = np.arange(len(err_labels))
    base_err = [float(results_df.iloc[0]["rmse"]), float(results_df.iloc[0]["mae"])]
    hyb_err = [float(results_df.iloc[1]["rmse"]), float(results_df.iloc[1]["mae"])]

    eb1 = ax2.bar(x2 - width / 2, base_err, width, label=results_df.iloc[0]["model"], color="#94a3b8", edgecolor="#475569")
    eb2 = ax2.bar(x2 + width / 2, hyb_err, width, label=results_df.iloc[1]["model"], color="#059669", edgecolor="#065f46")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(err_labels, fontweight="bold")
    ax2.set_ylim(0, max(base_err) * 1.25)
    ax2.set_title("Rating Error (Lower is Better)", fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(frameon=True, facecolor="white", fontsize=8.5)
    for b in [*eb1, *eb2]:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=8.5, xytext=(0, 3), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_report(
    results: list[dict[str, Any]],
    primary_k: int,
    sample_uid: int,
    sample_recs_df: pd.DataFrame,
    out_path: Path,
) -> None:
    r0, r1 = results[0], results[1]

    def _lift(col: str, lower_better: bool = False) -> str:
        v0, v1 = float(r0[col]), float(r1[col])
        if v0 == 0:
            return "N/A"
        pct = ((v0 - v1) / v0 * 100.0) if lower_better else ((v1 - v0) / v0 * 100.0)
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    lines = [
        "# Model Evaluation & Comparison Results",
        "",
        f"Primary cutoff: **K = {primary_k}** | Relevant item threshold: **rating >= 4.0**",
        "",
        f"## Side-by-Side Comparison (K = {primary_k})",
        "",
        f"| Metric | {r0['model']} | {r1['model']} | Relative Improvement |",
        "|---|---:|---:|---:|",
        f"| Precision@{primary_k} | {r0[f'precision_at_{primary_k}']:.4f} | **{r1[f'precision_at_{primary_k}']:.4f}** | {_lift(f'precision_at_{primary_k}')} |",
        f"| Recall@{primary_k} | {r0[f'recall_at_{primary_k}']:.4f} | **{r1[f'recall_at_{primary_k}']:.4f}** | {_lift(f'recall_at_{primary_k}')} |",
        f"| NDCG@{primary_k} | {r0[f'ndcg_at_{primary_k}']:.4f} | **{r1[f'ndcg_at_{primary_k}']:.4f}** | {_lift(f'ndcg_at_{primary_k}')} |",
        f"| Catalog Coverage@{primary_k} | {r0[f'coverage_at_{primary_k}']:.4f} | **{r1[f'coverage_at_{primary_k}']:.4f}** | {_lift(f'coverage_at_{primary_k}')} |",
        f"| Intra-List Diversity@{primary_k} | {r0[f'diversity_at_{primary_k}']:.4f} | **{r1[f'diversity_at_{primary_k}']:.4f}** | {_lift(f'diversity_at_{primary_k}')} |",
        f"| RMSE (lower is better) | {r0['rmse']:.4f} | **{r1['rmse']:.4f}** | {_lift('rmse', lower_better=True)} |",
        f"| MAE (lower is better) | {r0['mae']:.4f} | **{r1['mae']:.4f}** | {_lift('mae', lower_better=True)} |",
        "",
        f"## Sample Top-{primary_k} Hybrid Recommendations (User ID = {sample_uid})",
        "",
        "| Rank | Item ID | Title | Genres | Score |",
        "|---:|---:|---|---|---:|",
    ]
    for _, row in sample_recs_df.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {int(row['item_id'])} | {row['title']} | {row['genres']} | {row['score']:.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(config_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate trained models across ranking, beyond-accuracy, and error metrics."""
    configure_runtime_env()
    cfg = load_config(config_path)
    paths = ensure_directories(cfg)
    logger = get_logger("stage.evaluation")
    console = get_console()

    eval_cfg = cfg["evaluation"]
    k_values = [int(k) for k in eval_cfg["k_values"]]
    primary_k = int(eval_cfg["primary_k"])
    rel_thresh = float(eval_cfg["relevance_threshold"])
    sample_uid = int(eval_cfg["sample_user_id"])

    print_stage_banner(
        "Stage 7",
        "Model Evaluation & Comparison",
        f"Primary K={primary_k} | Relevance >= {rel_thresh} | Metrics: P@K, R@K, NDCG@K, Coverage, Diversity, RMSE, MAE",
    )

    proc_dir = paths["processed_dir"]
    models_dir = paths["models_dir"]
    rep_dir = paths["reports_dir"]

    test_df = pd.read_parquet(proc_dir / "test_interactions.parquet")
    items_df = pd.read_parquet(proc_dir / "items_clean.parquet")
    registry = json.loads((models_dir / "model_registry.json").read_text(encoding="utf-8"))

    with (models_dir / "popularity_recommender.pkl").open("rb") as handle:
        pop_model = pickle.load(handle)
    with (models_dir / "hybrid_svd_recommender.pkl").open("rb") as handle:
        hybrid_model = pickle.load(handle)

    genre_vecs = items_df[GENRE_COLUMNS].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(genre_vecs, axis=1, keepdims=True) + 1e-12
    genre_vecs_norm = genre_vecs / norms
    item_genre_norm_map = {
        int(iid): genre_vecs_norm[idx]
        for idx, iid in enumerate(items_df["item_id"].tolist())
    }
    all_item_ids = items_df["item_id"].astype(int).tolist()

    pop_metrics, _ = _evaluate_model(
        "Baseline Popularity",
        pop_model,
        test_df,
        all_item_ids,
        item_genre_norm_map,
        k_values,
        primary_k,
        rel_thresh,
    )
    hyb_metrics, hyb_recs = _evaluate_model(
        "Hybrid Personalized SVD",
        hybrid_model,
        test_df,
        all_item_ids,
        item_genre_norm_map,
        k_values,
        primary_k,
        rel_thresh,
    )

    results = [pop_metrics, hyb_metrics]
    results_df = pd.DataFrame(results)
    csv_path = rep_dir / "model_evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)

    chart_path = rep_dir / "model_comparison.png"
    _save_comparison_chart(results_df, primary_k, chart_path)

    # Sample user top-K table
    item_lookup = items_df.set_index("item_id")[["title", "genres"]]
    sample_rows = []
    for rank, (iid, score) in enumerate(hyb_recs[sample_uid][:primary_k], start=1):
        meta = item_lookup.loc[iid]
        sample_rows.append(
            {
                "rank": rank,
                "item_id": iid,
                "title": str(meta["title"]),
                "genres": str(meta["genres"]),
                "score": round(score, 4),
            }
        )
    sample_recs_df = pd.DataFrame(sample_rows)

    md_path = rep_dir / "model_evaluation_results.md"
    _write_markdown_report(results, primary_k, sample_uid, sample_recs_df, md_path)

    eval_payload = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_k": primary_k,
        "relevance_threshold": rel_thresh,
        "test_rows": len(test_df),
        "results": results,
    }
    (rep_dir / "model_evaluation_results.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")

    # Log evaluation metrics & plots back to MLflow runs
    mlflow.set_tracking_uri(_resolve_mlflow_uri(cfg["training"]["mlflow_tracking_uri"]))
    for key_name, metrics_dict in [
        ("baseline_popularity", pop_metrics),
        ("personalized_hybrid_svd", hyb_metrics),
    ]:
        run_id = registry["models"][key_name]["run_id"]
        with mlflow.start_run(run_id=run_id):
            numeric_metrics = {
                k: float(v)
                for k, v in metrics_dict.items()
                if isinstance(v, (int, float)) and k != "primary_k"
            }
            mlflow.log_metrics(numeric_metrics)
            mlflow.log_artifact(str(chart_path), artifact_path="evaluation")
            mlflow.log_artifact(str(csv_path), artifact_path="evaluation")

    comp_table = Table(title=f"Stage 7 Model Evaluation Comparison (K={primary_k})")
    comp_table.add_column("Model", style="cyan")
    comp_table.add_column(f"P@{primary_k}", justify="right", no_wrap=True)
    comp_table.add_column(f"R@{primary_k}", justify="right", no_wrap=True)
    comp_table.add_column(f"NDCG@{primary_k}", justify="right", no_wrap=True, style="bold green")
    comp_table.add_column(f"Cov@{primary_k}", justify="right", no_wrap=True)
    comp_table.add_column(f"Div@{primary_k}", justify="right", no_wrap=True)
    comp_table.add_column("RMSE", justify="right", no_wrap=True)
    comp_table.add_column("MAE", justify="right", no_wrap=True)

    for res in results:
        comp_table.add_row(
            res["model"],
            f"{res[f'precision_at_{primary_k}']:.4f}",
            f"{res[f'recall_at_{primary_k}']:.4f}",
            f"{res[f'ndcg_at_{primary_k}']:.4f}",
            f"{res[f'coverage_at_{primary_k}']:.4f}",
            f"{res[f'diversity_at_{primary_k}']:.4f}",
            f"{res['rmse']:.4f}",
            f"{res['mae']:.4f}",
        )
    console.print(comp_table)

    rec_table = Table(title=f"Sample Personalized Top-5 Recommendations (user_id={sample_uid})")
    rec_table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    rec_table.add_column("Movie Title", style="bold")
    rec_table.add_column("Genres", style="dim")
    rec_table.add_column("Hybrid Score", justify="right", style="green", no_wrap=True)
    for _, r in sample_recs_df.head(5).iterrows():
        rec_table.add_row(str(int(r["rank"])), str(r["title"]), str(r["genres"]), f"{r['score']:.4f}")
    console.print(rec_table)

    logger.info("Evaluation metrics logged to MLflow & saved to %s", md_path.relative_to(PROJECT_ROOT))
    return eval_payload


if __name__ == "__main__":
    run_evaluation()
