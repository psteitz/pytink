#!/usr/bin/env python3
"""
Backfill ``quote_start_date``/``quote_end_date`` into ``models.parquet``.

``ModelFarm._append_to_parquet`` now records the earliest/latest quote
timestamp actually used to train each model, but rows written before that
change lack these columns entirely. This script adds the columns (if
missing) and, for each row that lacks a value, tries to recover it from the
corresponding model's ``models/<TICKERS>/<TIMESTAMP>/config.yaml`` (written
by ``save_model`` under ``data.start_date``/``data.end_date``).

Important limitation: farm-trained models did not previously pass a quote
date range into ``save_model`` either, so their ``config.yaml`` files also
lack ``data.start_date``/``data.end_date``. The raw quote timestamps used
for those older runs were fetched from the database and discarded after
training -- they are not recoverable from anything on disk. This script
will therefore report those rows as un-backfillable rather than guess. It
exists so that any row whose model directory *does* carry the dates (e.g.
future runs, or models retrained with the updated code) gets picked up
automatically, and so the gap is reported explicitly instead of silently
ignored.

Usage:
    python -m pytink.backfill_quote_dates [--parquet-path PATH] [--models-dir PATH]
                                           [--tolerance-seconds N] [--dry-run]
"""
import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from pytink.model_registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_PARQUET_PATH,
    _TIMESTAMP_DIR_RE,
    _TIMESTAMP_FMT,
    _load_config_yaml,
)

logger = logging.getLogger(__name__)

# Candidate directory timestamp must be within this many seconds of the
# parquet row's created_at to be considered a match for that row.
DEFAULT_TOLERANCE_SECONDS = 3600

DirCandidate = Tuple[datetime, Optional[str], Optional[str]]  # (run_ts, start_date, end_date)


def _index_model_dirs(models_dir: Path) -> Dict[str, List[DirCandidate]]:
    """Scan ``models_dir`` and index each run's config by sorted-ticker key.

    Returns:
        Dict mapping ``"-".join(sorted(tickers))`` to a list of
        ``(run_timestamp, start_date, end_date)`` tuples, one per run
        directory found for that ticker set. ``start_date``/``end_date``
        are the raw ``data.start_date``/``data.end_date`` strings from
        ``config.yaml`` (``None`` when absent).
    """
    index: Dict[str, List[DirCandidate]] = {}
    if not models_dir.exists():
        logger.warning("Models directory not found: %s", models_dir)
        return index

    for ticker_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in ticker_dir.iterdir() if p.is_dir()):
            if not _TIMESTAMP_DIR_RE.match(run_dir.name):
                continue
            config_path = run_dir / "config.yaml"
            if not config_path.exists():
                continue
            config = _load_config_yaml(config_path)
            data = config.get("data", {}) or {}
            tickers = data.get("tickers") or ticker_dir.name.split("-")
            key = "-".join(sorted(tickers))
            run_ts = datetime.strptime(run_dir.name, _TIMESTAMP_FMT)
            index.setdefault(key, []).append(
                (run_ts, data.get("start_date"), data.get("end_date"))
            )
    return index


def backfill(
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
    dry_run: bool = False,
) -> dict:
    """Fill in missing ``quote_start_date``/``quote_end_date`` values in-place.

    Args:
        parquet_path: Path to ``models.parquet``.
        models_dir: Root ``models/`` directory to search for ``config.yaml``
            files carrying the quote date range.
        tolerance_seconds: Maximum allowed gap between a parquet row's
            ``created_at`` and a candidate directory's timestamp for that
            directory to be considered a match.
        dry_run: When ``True``, compute and report results without writing
            the parquet file.

    Returns:
        A summary dict with counts: ``total_rows``, ``already_present``,
        ``backfilled``, ``no_matching_dir``, ``no_dir_within_tolerance``,
        ``dir_found_no_dates_in_config``.
    """
    summary = {
        "total_rows": 0,
        "already_present": 0,
        "backfilled": 0,
        "no_matching_dir": 0,
        "no_dir_within_tolerance": 0,
        "dir_found_no_dates_in_config": 0,
    }

    if not parquet_path.exists():
        logger.error("Parquet file not found: %s", parquet_path)
        return summary

    df = pd.read_parquet(parquet_path)
    summary["total_rows"] = len(df)

    if "quote_start_date" not in df.columns:
        df["quote_start_date"] = pd.NaT
    if "quote_end_date" not in df.columns:
        df["quote_end_date"] = pd.NaT

    dir_index = _index_model_dirs(Path(models_dir))

    for idx, row in df.iterrows():
        if pd.notna(row["quote_start_date"]) and pd.notna(row["quote_end_date"]):
            summary["already_present"] += 1
            continue

        key = "-".join(sorted(str(row["tickers"]).split("-")))
        candidates = dir_index.get(key, [])
        if not candidates:
            summary["no_matching_dir"] += 1
            continue

        created_at = row["created_at"]
        run_ts, start_date, end_date = min(
            candidates, key=lambda c: abs((c[0] - created_at).total_seconds())
        )
        if abs((run_ts - created_at).total_seconds()) > tolerance_seconds:
            summary["no_dir_within_tolerance"] += 1
            continue
        if not start_date or not end_date:
            summary["dir_found_no_dates_in_config"] += 1
            continue

        df.at[idx, "quote_start_date"] = pd.Timestamp(start_date)
        df.at[idx, "quote_end_date"] = pd.Timestamp(end_date)
        summary["backfilled"] += 1

    if not dry_run and summary["backfilled"] > 0:
        backup_path = parquet_path.with_suffix(parquet_path.suffix + ".bak")
        shutil.copy(parquet_path, backup_path)
        df.to_parquet(parquet_path, index=False)
        logger.info("Backed up original to %s and wrote %s", backup_path, parquet_path)

    return summary


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Backfill quote_start_date/quote_end_date into models.parquet from config.yaml."
    )
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--tolerance-seconds", type=int, default=DEFAULT_TOLERANCE_SECONDS)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing the parquet file.",
    )
    args = parser.parse_args()

    summary = backfill(
        parquet_path=args.parquet_path,
        models_dir=args.models_dir,
        tolerance_seconds=args.tolerance_seconds,
        dry_run=args.dry_run,
    )

    logger.info("=" * 60)
    logger.info("Total rows:                      %d", summary["total_rows"])
    logger.info("Already had quote dates:         %d", summary["already_present"])
    logger.info("Backfilled from config.yaml:     %d", summary["backfilled"])
    logger.info("No matching model directory:     %d", summary["no_matching_dir"])
    logger.info("Matching dir outside tolerance:  %d", summary["no_dir_within_tolerance"])
    logger.info("Dir found but no dates in config:%d", summary["dir_found_no_dates_in_config"])
    logger.info("=" * 60)
    if summary["backfilled"] == 0:
        logger.info(
            "No rows were backfilled. This is expected for models trained "
            "before quote date tracking was added -- the raw quote "
            "timestamps they used were never persisted anywhere and cannot "
            "be recovered."
        )
    if args.dry_run:
        logger.info("Dry run: no files were modified.")


if __name__ == "__main__":
    main()
