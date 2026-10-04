#!/usr/bin/env python3
"""
Backfill the model registry (``models.parquet`` + per-model ``config.yaml``
files under ``models/``) in two ways:

1. **Missing parquet rows** -- only ``ModelFarm._append_to_parquet`` writes
   to ``models.parquet``, so models trained directly via ``pytink-train``
   (which only ever get a ``models/<TICKERS>/<TIMESTAMP>/`` directory with a
   ``training.log``) are invisible to anything that reads the parquet log.
   This adds a row for each such run, recovering metrics from
   ``training.log`` and hyperparameters from ``config.yaml``.
2. **``model_type``** -- every model pytink has ever trained is a GPT-2
   style causal LM (``AutoConfig.from_pretrained("gpt2")`` in
   :class:`~pytink.model.StockTransformerModel`), but nothing records that.
   Ahead of making the architecture configurable, this stamps
   ``model_type: "gpt2"`` into every model's ``config.yaml`` (under the
   ``model:`` section) and into every ``models.parquet`` row.

Usage:
    python -m pytink.backfill_registry [--parquet-path PATH] [--models-dir PATH]
                                        [--model-type gpt2] [--tolerance-seconds N]
                                        [--dry-run]
"""
import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

from pytink.model_registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_PARQUET_PATH,
    scan_models_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_TYPE = "gpt2"
DEFAULT_TOLERANCE_SECONDS = 3600

_CONFIG_HEADER = (
    "# Configuration used for this training run\n"
    "# Can be used with: python train_model.py --db-password PASSWORD --config config.yaml\n\n"
)


def _ticker_set(tickers_str: str) -> frozenset:
    return frozenset(str(tickers_str).split("-"))


def _backup(path: Path) -> None:
    """Copy *path* to ``<path>.bak`` if a backup doesn't already exist."""
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy(path, backup_path)


def add_missing_entries(
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    model_type: str = DEFAULT_MODEL_TYPE,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
    dry_run: bool = False,
) -> dict:
    """Append a ``models.parquet`` row for every on-disk run missing one.

    Only runs with a ``training.log`` (i.e. trained via ``pytink-train``,
    never logged by the farm) are considered. A run is treated as already
    present when an existing row has the same ticker set and a
    ``created_at`` within *tolerance_seconds* of the run's directory
    timestamp.

    Returns:
        Summary dict with ``added``, ``skipped_already_present``, and
        ``skipped_no_metrics`` counts.
    """
    summary = {"added": 0, "skipped_already_present": 0, "skipped_no_metrics": 0}

    df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    existing = []  # list of (ticker_set, created_at)
    if "tickers" in df.columns:
        created_ats = df["created_at"] if "created_at" in df.columns else [None] * len(df)
        existing = [(_ticker_set(t), ts) for t, ts in zip(df["tickers"], created_ats)]

    records = scan_models_dir(models_dir=models_dir, parquet_path=parquet_path)
    new_rows = []
    for record in records:
        if record.metrics_source != "training.log":
            continue  # Farm-trained runs already have a parquet row.
        if record.eval_accuracy is None or record.eval_loss is None:
            summary["skipped_no_metrics"] += 1
            continue

        key = frozenset(record.tickers)
        run_ts = record.run_timestamp
        already_present = any(
            key == existing_key
            and existing_ts is not None
            and abs((existing_ts - run_ts).total_seconds()) <= tolerance_seconds
            for existing_key, existing_ts in existing
        )
        if already_present:
            summary["skipped_already_present"] += 1
            continue

        new_rows.append({
            "tickers": record.ticker_key,
            "accuracy": record.eval_accuracy,
            "loss": record.eval_loss,
            "perplexity": record.perplexity,
            "interval_minutes": record.interval_minutes,
            "context_window_size": record.context_window_size,
            "batch_size": record.batch_size,
            "epochs": record.epochs,
            "learning_rate": record.learning_rate,
            "weight_decay": record.weight_decay,
            "early_stopping_patience": record.early_stopping_patience,
            "hidden_size": record.hidden_size,
            "num_hidden_layers": record.num_hidden_layers,
            "num_attention_heads": record.num_attention_heads,
            "max_position_embeddings": record.max_position_embeddings,
            "created_at": run_ts,
            "quote_start_date": pd.NaT,
            "quote_end_date": pd.NaT,
            "model_type": model_type,
        })
        existing.append((key, run_ts))  # Guard against duplicates within this run.

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        summary["added"] = len(new_rows)
        if not dry_run:
            if parquet_path.exists():
                _backup(parquet_path)
            df.to_parquet(parquet_path, index=False)

    return summary


def add_model_type(
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    model_type: str = DEFAULT_MODEL_TYPE,
    dry_run: bool = False,
) -> dict:
    """Stamp ``model_type`` into every ``models.parquet`` row and ``config.yaml``.

    Returns:
        Summary dict with ``parquet_rows_updated``, ``configs_updated``,
        ``configs_already_set``, and ``configs_missing`` counts.
    """
    summary = {
        "parquet_rows_updated": 0,
        "configs_updated": 0,
        "configs_already_set": 0,
        "configs_missing": 0,
    }

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if "model_type" not in df.columns:
            df["model_type"] = None
        mask = df["model_type"].isna()
        summary["parquet_rows_updated"] = int(mask.sum())
        if summary["parquet_rows_updated"] > 0:
            df.loc[mask, "model_type"] = model_type
            if not dry_run:
                _backup(parquet_path)
                df.to_parquet(parquet_path, index=False)

    if yaml is None:
        logger.warning("PyYAML not installed; cannot update config.yaml files.")
        return summary

    models_dir = Path(models_dir)
    if not models_dir.exists():
        return summary

    for ticker_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in ticker_dir.iterdir() if p.is_dir()):
            config_path = run_dir / "config.yaml"
            if not config_path.exists():
                summary["configs_missing"] += 1
                continue

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            model_section = config.get("model", {}) or {}
            if model_section.get("model_type") == model_type:
                summary["configs_already_set"] += 1
                continue

            # model_type goes first so it's the first thing read under `model:`.
            new_model_section = {"model_type": model_type}
            new_model_section.update(
                {k: v for k, v in model_section.items() if k != "model_type"}
            )
            config["model"] = new_model_section
            summary["configs_updated"] += 1

            if not dry_run:
                _backup(config_path)
                with open(config_path, "w") as f:
                    f.write(_CONFIG_HEADER)
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return summary


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Backfill missing models.parquet rows and model_type into the model registry."
    )
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--tolerance-seconds", type=int, default=DEFAULT_TOLERANCE_SECONDS)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing any files.",
    )
    args = parser.parse_args()

    missing_summary = add_missing_entries(
        parquet_path=args.parquet_path,
        models_dir=args.models_dir,
        model_type=args.model_type,
        tolerance_seconds=args.tolerance_seconds,
        dry_run=args.dry_run,
    )
    type_summary = add_model_type(
        parquet_path=args.parquet_path,
        models_dir=args.models_dir,
        model_type=args.model_type,
        dry_run=args.dry_run,
    )

    logger.info("=" * 60)
    logger.info("Step 1: add missing models.parquet rows from on-disk logs")
    logger.info("  Added:                   %d", missing_summary["added"])
    logger.info("  Already present:         %d", missing_summary["skipped_already_present"])
    logger.info("  Skipped (no metrics):    %d", missing_summary["skipped_no_metrics"])
    logger.info("-" * 60)
    logger.info("Step 2: stamp model_type=%r into the registry", args.model_type)
    logger.info("  models.parquet rows updated: %d", type_summary["parquet_rows_updated"])
    logger.info("  config.yaml files updated:   %d", type_summary["configs_updated"])
    logger.info("  config.yaml already set:     %d", type_summary["configs_already_set"])
    logger.info("  config.yaml missing:         %d", type_summary["configs_missing"])
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("Dry run: no files were modified.")


if __name__ == "__main__":
    main()
