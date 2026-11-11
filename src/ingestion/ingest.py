"""Stage 1: Data Ingestion module for MovieLens 100K dataset."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
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

GENRE_COLUMNS = [
    "genre_unknown",
    "genre_action",
    "genre_adventure",
    "genre_animation",
    "genre_childrens",
    "genre_comedy",
    "genre_crime",
    "genre_documentary",
    "genre_drama",
    "genre_fantasy",
    "genre_film_noir",
    "genre_horror",
    "genre_musical",
    "genre_mystery",
    "genre_romance",
    "genre_sci_fi",
    "genre_thriller",
    "genre_war",
    "genre_western",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_and_extract(url: str, raw_dir: Path, force_download: bool, logger: Any) -> Path:
    zip_path = raw_dir / "ml-100k.zip"
    extract_dir = raw_dir / "ml-100k"

    if force_download or not zip_path.exists():
        logger.info("Downloading MovieLens 100K from %s ...", url)
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        with zip_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    handle.write(chunk)
        logger.info("Saved archive to %s (%.2f MB)", zip_path.relative_to(PROJECT_ROOT), zip_path.stat().st_size / 1e6)
    else:
        logger.info("Using cached archive %s", zip_path.relative_to(PROJECT_ROOT))

    if force_download or not (extract_dir / "u.data").exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        logger.info("Extracted MovieLens files into %s", extract_dir.relative_to(PROJECT_ROOT))

    return extract_dir


def _build_audit_records(
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create documented dirty records so Stage 2 validation exercises every rule."""
    dirty_users = pd.DataFrame(
        [
            {"user_id": None, "age": 28, "gender": "M", "occupation": "engineer", "zip_code": "10001"},
            {"user_id": int(users_df.iloc[0]["user_id"]), "age": 24, "gender": "M", "occupation": "technician", "zip_code": "85711"},
            {"user_id": 9901, "age": "twenty", "gender": "F", "occupation": "student", "zip_code": "94105"},
            {"user_id": 9902, "age": 165, "gender": "X", "occupation": "artist", "zip_code": "30301"},
        ]
    )

    dirty_items = pd.DataFrame(
        [
            {
                "item_id": 99001,
                "title": "",
                "release_date": "01-Jan-1996",
                "video_release_date": None,
                "imdb_url": "http://example.com/missing",
                **{g: 0 for g in GENRE_COLUMNS},
                "genres": "unknown",
            },
            {
                **items_df.iloc[0].to_dict(),
            },
            {
                "item_id": "bad_id",
                "title": "Corrupt Movie ID",
                "release_date": "01-Jan-1997",
                "video_release_date": None,
                "imdb_url": "",
                **{g: 0 for g in GENRE_COLUMNS},
                "genres": "Drama",
            },
        ]
    )

    first_row = interactions_df.iloc[0]
    dirty_interactions = pd.DataFrame(
        [
            {"user_id": None, "item_id": 50, "rating": 4, "timestamp": 881250949, "event_datetime": "1997-12-04T15:55:49+00:00"},
            {"user_id": 10, "item_id": 50, "rating": None, "timestamp": 881250949, "event_datetime": "1997-12-04T15:55:49+00:00"},
            {
                "user_id": int(first_row["user_id"]),
                "item_id": int(first_row["item_id"]),
                "rating": int(first_row["rating"]),
                "timestamp": int(first_row["timestamp"]),
                "event_datetime": first_row["event_datetime"],
            },
            {"user_id": 12, "item_id": 100, "rating": "five", "timestamp": 881250949, "event_datetime": "1997-12-04T15:55:49+00:00"},
            {"user_id": 15, "item_id": 120, "rating": 0, "timestamp": 881250949, "event_datetime": "1997-12-04T15:55:49+00:00"},
            {"user_id": 18, "item_id": 125, "rating": 6, "timestamp": 881250949, "event_datetime": "1997-12-04T15:55:49+00:00"},
            {"user_id": 20, "item_id": 130, "rating": 4, "timestamp": -100, "event_datetime": "invalid"},
            {"user_id": 22, "item_id": 140, "rating": 5, "timestamp": 4102444800, "event_datetime": "2100-01-01T00:00:00+00:00"},
            {"user_id": 99999, "item_id": 150, "rating": 4, "timestamp": 881250949, "event_datetime": "1997-12-04T15:55:49+00:00"},
            {"user_id": 25, "item_id": 88888, "rating": 3, "timestamp": 881250949, "event_datetime": "1997-12-04T15:55:49+00:00"},
        ]
    )
    return dirty_users, dirty_items, dirty_interactions


def run_ingestion(config_path: str | Path | None = None) -> dict[str, Any]:
    """Ingest MovieLens 100K users, items, and interactions into raw CSV staging files."""
    configure_runtime_env()
    cfg = load_config(config_path)
    paths = ensure_directories(cfg)
    logger = get_logger("stage.ingestion")
    console = get_console()

    print_stage_banner(
        "Stage 1",
        "Data Ingestion",
        f"Source: {cfg['ingestion']['dataset_name']} | Staging Dir: {cfg['paths']['raw_dir']}",
    )

    raw_dir = paths["raw_dir"]
    extract_dir = _download_and_extract(
        url=cfg["ingestion"]["dataset_url"],
        raw_dir=raw_dir,
        force_download=bool(cfg["ingestion"].get("force_download", False)),
        logger=logger,
    )

    users_df = pd.read_csv(
        extract_dir / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )

    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url", *GENRE_COLUMNS]
    items_df = pd.read_csv(
        extract_dir / "u.item",
        sep="|",
        names=item_cols,
        encoding="latin-1",
    )
    genre_labels = [col.replace("genre_", "").replace("_", " ").title() for col in GENRE_COLUMNS]
    items_df["genres"] = items_df[GENRE_COLUMNS].apply(
        lambda row: "|".join(lbl for lbl, flag in zip(genre_labels, row) if flag == 1) or "Unknown",
        axis=1,
    )

    interactions_df = pd.read_csv(
        extract_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    interactions_df["event_datetime"] = pd.to_datetime(
        interactions_df["timestamp"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    source_counts = {
        "users": len(users_df),
        "items": len(items_df),
        "interactions": len(interactions_df),
    }

    injected_counts = {"users": 0, "items": 0, "interactions": 0}
    if cfg["ingestion"].get("inject_validation_test_records", False):
        dirty_u, dirty_i, dirty_inter = _build_audit_records(users_df, items_df, interactions_df)
        injected_counts = {
            "users": len(dirty_u),
            "items": len(dirty_i),
            "interactions": len(dirty_inter),
        }
        users_df = pd.concat([users_df.astype(object), dirty_u], ignore_index=True)
        items_df = pd.concat([items_df.astype(object), dirty_i], ignore_index=True)
        interactions_df = pd.concat([interactions_df.astype(object), dirty_inter], ignore_index=True)
        logger.info(
            "Appended %d validation audit records (users=%d, items=%d, interactions=%d)",
            sum(injected_counts.values()),
            injected_counts["users"],
            injected_counts["items"],
            injected_counts["interactions"],
        )

    users_out = raw_dir / "users.csv"
    items_out = raw_dir / "items.csv"
    interactions_out = raw_dir / "interactions.csv"

    users_df.to_csv(users_out, index=False)
    items_df.to_csv(items_out, index=False)
    interactions_df.to_csv(interactions_out, index=False)

    manifest = {
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": cfg["ingestion"]["dataset_name"],
        "source_url": cfg["ingestion"]["dataset_url"],
        "archive_sha256": _sha256_file(raw_dir / "ml-100k.zip"),
        "source_counts": source_counts,
        "injected_audit_counts": injected_counts,
        "raw_output_counts": {
            "users": len(users_df),
            "items": len(items_df),
            "interactions": len(interactions_df),
        },
        "files": {
            "users": str(users_out.relative_to(PROJECT_ROOT)),
            "items": str(items_out.relative_to(PROJECT_ROOT)),
            "interactions": str(interactions_out.relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path = raw_dir / "ingestion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    table = Table(title="Stage 1 Ingestion Summary", show_lines=False)
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Source Rows", justify="right", no_wrap=True)
    table.add_column("Audit Test Rows", justify="right", no_wrap=True)
    table.add_column("Staged Raw Rows", justify="right", style="bold green", no_wrap=True)
    table.add_column("Size (KB)", justify="right", no_wrap=True)
    table.add_column("Output Path", style="dim", no_wrap=True)

    for name, out_file in [
        ("users", users_out),
        ("items", items_out),
        ("interactions", interactions_out),
    ]:
        table.add_row(
            name,
            f"{source_counts[name]:,}",
            f"{injected_counts[name]:,}",
            f"{manifest['raw_output_counts'][name]:,}",
            f"{out_file.stat().st_size / 1024:.1f}",
            str(out_file.relative_to(PROJECT_ROOT)),
        )
    console.print(table)
    logger.info("Ingestion manifest written to %s", manifest_path.relative_to(PROJECT_ROOT))
    return manifest


if __name__ == "__main__":
    run_ingestion()
