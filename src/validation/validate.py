"""Stage 2: Automated Data Validation, Pandera Contract Verification & Quarantine."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema
from rich.table import Table

from src.utils.config import (
    PROJECT_ROOT,
    configure_runtime_env,
    ensure_directories,
    get_console,
    get_logger,
    load_config,
    print_stage_banner,
)


def _is_int_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse values into integers cleanly, returning (parsed_numeric, is_valid_int_mask)."""
    numeric = pd.to_numeric(series, errors="coerce")
    is_valid = numeric.notna() & np.isfinite(numeric) & (numeric % 1 == 0)
    return numeric, is_valid


def _quarantine_rows(
    df: pd.DataFrame,
    invalid_reasons: dict[int, tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into clean rows and quarantined rows annotated with error metadata."""
    if not invalid_reasons:
        empty_err = df.iloc[0:0].copy()
        empty_err["error_code"] = pd.Series(dtype="string")
        empty_err["error_reason"] = pd.Series(dtype="string")
        return df.copy(), empty_err

    bad_idx = sorted(invalid_reasons.keys())
    err_df = df.loc[bad_idx].copy()
    err_df["error_code"] = [invalid_reasons[i][0] for i in bad_idx]
    err_df["error_reason"] = [invalid_reasons[i][1] for i in bad_idx]
    clean_df = df.drop(index=bad_idx).copy()
    return clean_df, err_df


def _record_error(
    reasons: dict[int, tuple[str, str]],
    mask: pd.Series,
    code: str,
    reason: str,
) -> int:
    """Assign error code/reason to rows matching mask if not already flagged."""
    count = 0
    for idx in mask[mask].index:
        if idx not in reasons:
            reasons[idx] = (code, reason)
            count += 1
    return count


def _validate_users(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    val_cfg = cfg["validation"]
    reasons: dict[int, tuple[str, str]] = {}
    counts: dict[str, int] = {}

    missing_mask = df[["user_id", "age", "gender", "occupation"]].isna().any(axis=1)
    counts["missing_required_fields"] = _record_error(reasons, missing_mask, "MISSING_FIELD", "Null user_id/age/gender/occupation")

    uid_num, uid_ok = _is_int_series(df["user_id"])
    age_num, age_ok = _is_int_series(df["age"])
    type_bad = (~uid_ok | ~age_ok) & ~missing_mask
    counts["invalid_data_types"] = _record_error(reasons, type_bad, "INVALID_TYPE", "Non-integer user_id or age")

    range_bad = (
        uid_ok
        & age_ok
        & ((age_num < val_cfg["age_min"]) | (age_num > val_cfg["age_max"]) | ~df["gender"].astype(str).str.upper().isin(val_cfg["valid_genders"]))
    )
    counts["domain_range_violations"] = _record_error(
        reasons,
        range_bad,
        "DOMAIN_VIOLATION",
        f"Age outside [{val_cfg['age_min']}, {val_cfg['age_max']}] or invalid gender",
    )

    candidate_clean = uid_ok & ~pd.Series(df.index.isin(reasons.keys()), index=df.index)
    dup_mask = pd.Series(False, index=df.index)
    dup_mask.loc[candidate_clean] = uid_num.loc[candidate_clean].duplicated(keep="first")
    counts["duplicate_primary_keys"] = _record_error(reasons, dup_mask, "DUPLICATE_PK", "Duplicate user_id")

    clean_df, err_df = _quarantine_rows(df, reasons)
    clean_df["user_id"] = pd.to_numeric(clean_df["user_id"]).astype("int64")
    clean_df["age"] = pd.to_numeric(clean_df["age"]).astype("int64")
    clean_df["gender"] = clean_df["gender"].astype(str).str.strip().str.upper()
    clean_df["occupation"] = clean_df["occupation"].astype(str).str.strip().str.lower()
    clean_df["zip_code"] = clean_df["zip_code"].fillna("00000").astype(str).str.strip()
    return clean_df, err_df, counts


def _validate_items(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    reasons: dict[int, tuple[str, str]] = {}
    counts: dict[str, int] = {}

    title_missing = df["title"].isna() | (df["title"].astype(str).str.strip() == "")
    id_missing = df["item_id"].isna()
    missing_mask = title_missing | id_missing
    counts["missing_required_fields"] = _record_error(reasons, missing_mask, "MISSING_FIELD", "Null item_id or empty title")

    iid_num, iid_ok = _is_int_series(df["item_id"])
    type_bad = ~iid_ok & ~missing_mask
    counts["invalid_data_types"] = _record_error(reasons, type_bad, "INVALID_TYPE", "Non-integer item_id")

    candidate_clean = iid_ok & ~pd.Series(df.index.isin(reasons.keys()), index=df.index)
    dup_mask = pd.Series(False, index=df.index)
    dup_mask.loc[candidate_clean] = iid_num.loc[candidate_clean].duplicated(keep="first")
    counts["duplicate_primary_keys"] = _record_error(reasons, dup_mask, "DUPLICATE_PK", "Duplicate item_id")

    clean_df, err_df = _quarantine_rows(df, reasons)
    clean_df["item_id"] = pd.to_numeric(clean_df["item_id"]).astype("int64")
    genre_cols = [c for c in clean_df.columns if c.startswith("genre_")]
    for gcol in genre_cols:
        clean_df[gcol] = pd.to_numeric(clean_df[gcol], errors="coerce").fillna(0).astype("int64")
    return clean_df, err_df, counts


def _validate_interactions(
    df: pd.DataFrame,
    valid_user_ids: set[int],
    valid_item_ids: set[int],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    val_cfg = cfg["validation"]
    reasons: dict[int, tuple[str, str]] = {}
    counts: dict[str, int] = {}

    missing_mask = df[["user_id", "item_id", "rating", "timestamp"]].isna().any(axis=1)
    counts["missing_required_fields"] = _record_error(reasons, missing_mask, "MISSING_FIELD", "Null user_id/item_id/rating/timestamp")

    uid_num, uid_ok = _is_int_series(df["user_id"])
    iid_num, iid_ok = _is_int_series(df["item_id"])
    ts_num, ts_ok = _is_int_series(df["timestamp"])
    rating_num = pd.to_numeric(df["rating"], errors="coerce")
    rating_ok = rating_num.notna() & np.isfinite(rating_num)

    type_bad = (~uid_ok | ~iid_ok | ~ts_ok | ~rating_ok) & ~missing_mask
    counts["invalid_data_types"] = _record_error(reasons, type_bad, "INVALID_TYPE", "Non-numeric user_id/item_id/rating/timestamp")

    all_types_ok = uid_ok & iid_ok & ts_ok & rating_ok
    rating_bad = all_types_ok & ((rating_num < val_cfg["rating_min"]) | (rating_num > val_cfg["rating_max"]))
    counts["invalid_ratings"] = _record_error(
        reasons,
        rating_bad,
        "INVALID_RATING",
        f"Rating outside [{val_cfg['rating_min']}, {val_cfg['rating_max']}]",
    )

    ts_bad = all_types_ok & ~rating_bad & ((ts_num < val_cfg["min_timestamp"]) | (ts_num > val_cfg["max_timestamp"]))
    counts["invalid_timestamps"] = _record_error(
        reasons,
        ts_bad,
        "INVALID_TIMESTAMP",
        f"Timestamp outside [{val_cfg['min_timestamp']}, {val_cfg['max_timestamp']}]",
    )

    not_flagged = ~pd.Series(df.index.isin(reasons.keys()), index=df.index)
    ref_bad = (
        not_flagged
        & (~uid_num.isin(valid_user_ids) | ~iid_num.isin(valid_item_ids))
    )
    counts["referential_integrity_violations"] = _record_error(
        reasons,
        ref_bad,
        "REFERENTIAL_INTEGRITY",
        "Orphan user_id or item_id missing from master tables",
    )

    still_ok = ~pd.Series(df.index.isin(reasons.keys()), index=df.index)
    dup_pair = pd.Series(False, index=df.index)
    pair_frame = pd.DataFrame({"u": uid_num.loc[still_ok], "i": iid_num.loc[still_ok]})
    dup_pair.loc[still_ok] = pair_frame.duplicated(subset=["u", "i"], keep="first")
    counts["duplicate_interactions"] = _record_error(
        reasons,
        dup_pair,
        "DUPLICATE_INTERACTION",
        "Duplicate (user_id, item_id) interaction record",
    )

    clean_df, err_df = _quarantine_rows(df, reasons)
    clean_df["user_id"] = pd.to_numeric(clean_df["user_id"]).astype("int64")
    clean_df["item_id"] = pd.to_numeric(clean_df["item_id"]).astype("int64")
    clean_df["rating"] = pd.to_numeric(clean_df["rating"]).astype("float64")
    clean_df["timestamp"] = pd.to_numeric(clean_df["timestamp"]).astype("int64")
    return clean_df, err_df, counts


def _verify_pandera_contracts(
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, str]:
    """Run formal Pandera DataFrameSchema assertions on post-quarantine data."""
    val_cfg = cfg["validation"]

    user_schema = DataFrameSchema(
        {
            "user_id": Column(int, checks=[Check.gt(0)], unique=True, nullable=False),
            "age": Column(int, checks=[Check.in_range(val_cfg["age_min"], val_cfg["age_max"])], nullable=False),
            "gender": Column(str, checks=[Check.isin(val_cfg["valid_genders"])], nullable=False),
            "occupation": Column(str, checks=[Check.str_length(min_value=1)], nullable=False),
        },
        strict=False,
    )
    item_schema = DataFrameSchema(
        {
            "item_id": Column(int, checks=[Check.gt(0)], unique=True, nullable=False),
            "title": Column(str, checks=[Check.str_length(min_value=1)], nullable=False),
            "genres": Column(str, nullable=False),
        },
        strict=False,
    )
    interaction_schema = DataFrameSchema(
        {
            "user_id": Column(int, checks=[Check.isin(set(users_df["user_id"]))], nullable=False),
            "item_id": Column(int, checks=[Check.isin(set(items_df["item_id"]))], nullable=False),
            "rating": Column(float, checks=[Check.in_range(val_cfg["rating_min"], val_cfg["rating_max"])], nullable=False),
            "timestamp": Column(
                int,
                checks=[Check.in_range(val_cfg["min_timestamp"], val_cfg["max_timestamp"])],
                nullable=False,
            ),
        },
        unique=["user_id", "item_id"],
        strict=False,
    )

    user_schema.validate(users_df, lazy=True)
    item_schema.validate(items_df, lazy=True)
    interaction_schema.validate(interactions_df, lazy=True)
    return {
        "users_schema": "PASSED (4 column contracts, PK uniqueness)",
        "items_schema": "PASSED (3 column contracts, PK uniqueness)",
        "interactions_schema": "PASSED (4 column contracts, composite PK & FK integrity)",
    }


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Data Validation & Quality Report",
        "",
        f"- **Generated (UTC)**: `{report['validated_at_utc']}`",
        f"- **Overall Validation Gate**: `{report['gate_status']}`",
        f"- **Pandera Contracts**: `ALL PASSED`",
        "",
        "## Entity Validation Summary",
        "",
        "| Dataset | Raw Rows | Clean Rows | Quarantined Rows | Error Rate (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for entity, stats in report["entities"].items():
        lines.append(
            f"| `{entity}` | {stats['raw_rows']:,} | {stats['clean_rows']:,} | "
            f"{stats['quarantined_rows']:,} | {stats['error_rate_pct']:.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Check Breakdown by Failure Mode",
            "",
            "| Dataset | Check Category | Quarantined Records |",
            "|---|---|---:|",
        ]
    )
    for entity, stats in report["entities"].items():
        for check_name, count in stats["checks"].items():
            lines.append(f"| `{entity}` | `{check_name}` | {count} |")

    lines.extend(
        [
            "",
            "## Non-Blocking Source Null Audit (Items Metadata)",
            "",
            f"- `release_date` missing rows (handled in Preparation): `{report['source_null_audit']['items_release_date_nulls']}`",
            f"- `video_release_date` missing rows (dropped in Preparation): `{report['source_null_audit']['items_video_release_date_nulls']}`",
            f"- `imdb_url` missing rows: `{report['source_null_audit']['items_imdb_url_nulls']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(config_path: str | Path | None = None) -> dict[str, Any]:
    """Validate raw CSV datasets, quarantine invalid records, and verify Pandera schemas."""
    configure_runtime_env()
    cfg = load_config(config_path)
    paths = ensure_directories(cfg)
    logger = get_logger("stage.validation")
    console = get_console()

    print_stage_banner(
        "Stage 2",
        "Data Validation & Quarantine",
        f"Input: {cfg['paths']['raw_dir']} | Errors: {cfg['paths']['errors_dir']}",
    )

    raw_dir = paths["raw_dir"]
    err_dir = paths["errors_dir"]
    proc_dir = paths["processed_dir"]
    rep_dir = paths["reports_dir"]

    users_raw = pd.read_csv(raw_dir / "users.csv", dtype=str, keep_default_na=True)
    items_raw = pd.read_csv(raw_dir / "items.csv", dtype=str, keep_default_na=True)
    interactions_raw = pd.read_csv(raw_dir / "interactions.csv", dtype=str, keep_default_na=True)

    clean_users, bad_users, user_checks = _validate_users(users_raw, cfg)
    clean_items, bad_items, item_checks = _validate_items(items_raw)
    clean_interactions, bad_interactions, inter_checks = _validate_interactions(
        interactions_raw,
        valid_user_ids=set(clean_users["user_id"]),
        valid_item_ids=set(clean_items["item_id"]),
        cfg=cfg,
    )

    contract_status = _verify_pandera_contracts(clean_users, clean_items, clean_interactions, cfg)
    logger.info("Pandera schema verification passed across all 3 validated datasets")

    bad_users.to_csv(err_dir / "invalid_users.csv", index=False)
    bad_items.to_csv(err_dir / "invalid_items.csv", index=False)
    bad_interactions.to_csv(err_dir / "invalid_interactions.csv", index=False)

    clean_users.to_parquet(proc_dir / "validated_users.parquet", index=False)
    clean_items.to_parquet(proc_dir / "validated_items.parquet", index=False)
    clean_interactions.to_parquet(proc_dir / "validated_interactions.parquet", index=False)

    def _entity_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame, bad_df: pd.DataFrame, checks: dict[str, int]) -> dict[str, Any]:
        rate = (len(bad_df) / max(len(raw_df), 1)) * 100.0
        return {
            "raw_rows": len(raw_df),
            "clean_rows": len(clean_df),
            "quarantined_rows": len(bad_df),
            "error_rate_pct": round(rate, 4),
            "checks": checks,
        }

    entities_report = {
        "users": _entity_summary(users_raw, clean_users, bad_users, user_checks),
        "items": _entity_summary(items_raw, clean_items, bad_items, item_checks),
        "interactions": _entity_summary(interactions_raw, clean_interactions, bad_interactions, inter_checks),
    }

    max_err_pct = float(cfg["validation"]["max_error_rate_pct"])
    gate_passed = all(v["error_rate_pct"] <= max_err_pct for v in entities_report.values())

    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_status": "PASSED" if gate_passed else "FAILED",
        "max_error_rate_pct": max_err_pct,
        "entities": entities_report,
        "pandera_contracts": contract_status,
        "source_null_audit": {
            "items_release_date_nulls": int(clean_items["release_date"].isna().sum()),
            "items_video_release_date_nulls": int(clean_items["video_release_date"].isna().sum()),
            "items_imdb_url_nulls": int(clean_items["imdb_url"].isna().sum()),
        },
    }

    report_json_path = rep_dir / "data_validation_report.json"
    report_md_path = rep_dir / "data_validation_report.md"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_report(report, report_md_path)

    table = Table(title="Stage 2 Validation & Quarantine Summary")
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Raw Rows", justify="right")
    table.add_column("Clean Rows", justify="right", style="bold green")
    table.add_column("Quarantined", justify="right", style="bold yellow")
    table.add_column("Error Rate", justify="right")
    table.add_column("Pandera Contract", style="green")

    for name in ("users", "items", "interactions"):
        st = entities_report[name]
        table.add_row(
            name,
            f"{st['raw_rows']:,}",
            f"{st['clean_rows']:,}",
            f"{st['quarantined_rows']:,}",
            f"{st['error_rate_pct']:.3f}%",
            "PASSED",
        )
    console.print(table)

    err_table = Table(title="Quarantined Anomaly Breakdown (Interactions)")
    err_table.add_column("Error Code", style="yellow", no_wrap=True)
    err_table.add_column("Count", justify="right")
    err_table.add_column("Sample Reason", style="dim")
    if not bad_interactions.empty:
        grouped = bad_interactions.groupby("error_code", sort=False).agg(
            count=("error_code", "size"),
            reason=("error_reason", "first"),
        )
        for code, row in grouped.iterrows():
            err_table.add_row(str(code), str(int(row["count"])), str(row["reason"]))
    console.print(err_table)
    logger.info("Validation report saved to %s", report_md_path.relative_to(PROJECT_ROOT))

    if not gate_passed:
        raise ValueError(f"Data quality gate failed: error rate exceeded {max_err_pct}%")

    return report


if __name__ == "__main__":
    run_validation()

