"""Scans the ``models/`` directory tree and the farm's ``models.parquet``
log to build a unified, queryable registry of trained pytink models.

Each trained model lives at ``models/<TICKERS>/<TIMESTAMP>/`` and contains
``config.yaml`` and ``model.pt``, and -- for models trained via
``pytink.train_model`` -- a ``training.log`` with final evaluation metrics
and per-stock accuracy. Models trained via ``pytink.farming`` do not carry a
``training.log``; their metrics instead live in the root-level
``models.parquet`` file, keyed by ticker set and creation timestamp.

This module contains only data classes and module-level functions; it has no
UI dependencies so it can be used and tested independently of the viewer.

Exported classes:
    ModelRecord -- A single trained model's metadata, config, and metrics.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DEFAULT_PARQUET_PATH = Path(__file__).parent.parent.parent / "models.parquet"

_TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}$")
_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"

_FINAL_LOSS_RE = re.compile(r"Final Eval Loss:\s*([\d.eE+-]+)")
_FINAL_ACCURACY_RE = re.compile(r"Final Eval Accuracy:\s*([\d.eE+-]+)")
_FINAL_PERPLEXITY_RE = re.compile(r"Final Perplexity:\s*([\d.eE+-]+)")
_PER_STOCK_RE = re.compile(r"^([A-Za-z.]+) \(position \d+/\d+\) - Accuracy:\s*([\d.]+)")
_UNIQUE_WORDS_RE = re.compile(r"Unique words:\s*(\d+)")
_TOTAL_WORDS_RE = re.compile(r"Generated\s*(\d+)\s*words")
_SEQUENCES_RE = re.compile(r"Train sequences:\s*(\d+),\s*Eval sequences:\s*(\d+)")
_TRAIN_DURATION_RE = re.compile(r"Model training completed in\s*([\d.]+) seconds")


@dataclass
class ModelRecord:
    """A single trained model: its location, configuration, and metrics."""

    tickers: List[str]
    model_dir: Path
    run_timestamp: Optional[datetime]
    raw_config: dict = field(default_factory=dict)
    eval_accuracy: Optional[float] = None
    eval_loss: Optional[float] = None
    perplexity: Optional[float] = None
    metrics_source: str = "none"  # "training.log" | "models.parquet" | "none"
    per_stock_accuracy: Dict[str, float] = field(default_factory=dict)
    vocab_size: Optional[int] = None
    num_words: Optional[int] = None
    train_sequences: Optional[int] = None
    eval_sequences: Optional[int] = None
    training_seconds: Optional[float] = None

    @property
    def ticker_key(self) -> str:
        """Directory-style ticker key, e.g. ``'AAPL-GOOGL-MSFT'``."""
        return "-".join(self.tickers)

    def _section(self, name: str) -> dict:
        return self.raw_config.get(name, {}) or {}

    @property
    def interval_minutes(self):
        return self._section("data").get("interval_minutes")

    @property
    def context_window_size(self):
        data = self._section("data")
        return data.get("context_window_size", data.get("sequence_length"))

    @property
    def hidden_size(self):
        return self._section("model").get("hidden_size")

    @property
    def num_hidden_layers(self):
        return self._section("model").get("num_hidden_layers")

    @property
    def num_attention_heads(self):
        return self._section("model").get("num_attention_heads")

    @property
    def max_position_embeddings(self):
        return self._section("model").get("max_position_embeddings")

    @property
    def batch_size(self):
        return self._section("training").get("batch_size")

    @property
    def epochs(self):
        return self._section("training").get("num_epochs")

    @property
    def learning_rate(self):
        return self._section("training").get("learning_rate")

    @property
    def weight_decay(self):
        return self._section("training").get("weight_decay")

    @property
    def early_stopping_patience(self):
        return self._section("training").get("early_stopping_patience")

    def to_flat_dict(self) -> dict:
        """Flatten this record into a single-level dict for a DataFrame row."""
        return {
            "tickers": list(self.tickers),
            "tickers_display": self.ticker_key,
            "model_dir": str(self.model_dir),
            "run_timestamp": self.run_timestamp,
            "eval_accuracy": self.eval_accuracy,
            "eval_loss": self.eval_loss,
            "perplexity": self.perplexity,
            "metrics_source": self.metrics_source,
            "interval_minutes": self.interval_minutes,
            "context_window_size": self.context_window_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "vocab_size": self.vocab_size,
            "num_words": self.num_words,
            "train_sequences": self.train_sequences,
            "eval_sequences": self.eval_sequences,
            "training_seconds": self.training_seconds,
            "per_stock_accuracy": dict(self.per_stock_accuracy),
            "raw_config": self.raw_config,
        }


def _load_config_yaml(config_path: Path) -> dict:
    """Load a model's ``config.yaml``, returning ``{}`` on any failure."""
    if yaml is None:
        logger.warning("PyYAML not installed; cannot parse %s", config_path)
        return {}
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to parse %s: %s", config_path, e)
        return {}


def _parse_training_log(log_path: Path) -> dict:
    """Extract final metrics and per-stock accuracy from a ``training.log``.

    Returns a dict with keys matching the metric fields of :class:`ModelRecord`
    (only those that were found in the log are present).
    """
    result: dict = {}
    per_stock: Dict[str, float] = {}
    try:
        text = log_path.read_text(errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", log_path, e)
        return result

    for line in text.splitlines():
        m = _FINAL_LOSS_RE.search(line)
        if m:
            result["eval_loss"] = float(m.group(1))
            continue
        m = _FINAL_ACCURACY_RE.search(line)
        if m:
            result["eval_accuracy"] = float(m.group(1))
            continue
        m = _FINAL_PERPLEXITY_RE.search(line)
        if m:
            result["perplexity"] = float(m.group(1))
            continue
        m = _UNIQUE_WORDS_RE.search(line)
        if m:
            result["vocab_size"] = int(m.group(1))
            continue
        m = _TOTAL_WORDS_RE.search(line)
        if m:
            result["num_words"] = int(m.group(1))
            continue
        m = _SEQUENCES_RE.search(line)
        if m:
            result["train_sequences"] = int(m.group(1))
            result["eval_sequences"] = int(m.group(2))
            continue
        m = _TRAIN_DURATION_RE.search(line)
        if m:
            result["training_seconds"] = float(m.group(1))
            continue
        m = _PER_STOCK_RE.search(line)
        if m:
            per_stock[m.group(1)] = float(m.group(2))
            continue

    if per_stock:
        result["per_stock_accuracy"] = per_stock
    return result


class _ParquetIndex:
    """Looks up farm-trained models' metrics by ticker set and timestamp."""

    def __init__(self, parquet_path: Path):
        self._df: Optional[pd.DataFrame] = None
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                df["_ticker_set"] = df["tickers"].apply(
                    lambda s: frozenset(s.split("-"))
                )
                self._df = df
            except Exception as e:
                logger.warning("Failed to load %s: %s", parquet_path, e)

    def lookup(self, tickers: List[str], run_timestamp: Optional[datetime]) -> Optional[dict]:
        """Return the metrics dict for the closest-matching parquet row, if any."""
        if self._df is None or self._df.empty:
            return None
        target = frozenset(tickers)
        candidates = self._df[self._df["_ticker_set"] == target]
        if candidates.empty:
            return None
        if run_timestamp is not None and len(candidates) > 1:
            diffs = (candidates["created_at"] - run_timestamp).abs()
            row = candidates.loc[diffs.idxmin()]
        else:
            row = candidates.iloc[0]
        return {
            "eval_accuracy": float(row["accuracy"]),
            "eval_loss": float(row["loss"]),
            "perplexity": float(row["perplexity"]),
        }


def scan_models_dir(
    models_dir: Optional[Path] = None,
    parquet_path: Optional[Path] = None,
) -> List[ModelRecord]:
    """Scan ``models_dir`` and return a :class:`ModelRecord` for each run found.

    Ticker directories that do not contain a properly named
    ``<TIMESTAMP>`` subdirectory (``YYYYMMDD_HHMMSS``) with a ``config.yaml``
    are silently skipped -- these are legacy runs predating the dated
    subdirectory layout.

    Args:
        models_dir: Root models directory (default: ``<repo>/models``).
        parquet_path: Path to the farm's aggregated metrics log
            (default: ``<repo>/models.parquet``). Used to fill in metrics
            for models that lack a ``training.log``.

    Returns:
        List of :class:`ModelRecord`, one per discovered training run.
    """
    models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    parquet_path = Path(parquet_path) if parquet_path else DEFAULT_PARQUET_PATH

    if not models_dir.exists():
        logger.warning("Models directory not found: %s", models_dir)
        return []

    parquet_index = _ParquetIndex(parquet_path)
    records: List[ModelRecord] = []

    for ticker_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in ticker_dir.iterdir() if p.is_dir()):
            if not _TIMESTAMP_DIR_RE.match(run_dir.name):
                continue
            config_path = run_dir / "config.yaml"
            if not config_path.exists():
                continue

            config = _load_config_yaml(config_path)
            tickers = config.get("data", {}).get("tickers") or ticker_dir.name.split("-")
            run_timestamp = datetime.strptime(run_dir.name, _TIMESTAMP_FMT)

            record = ModelRecord(
                tickers=list(tickers),
                model_dir=run_dir,
                run_timestamp=run_timestamp,
                raw_config=config,
            )

            log_path = run_dir / "training.log"
            if log_path.exists():
                metrics = _parse_training_log(log_path)
                if metrics:
                    record.eval_accuracy = metrics.get("eval_accuracy")
                    record.eval_loss = metrics.get("eval_loss")
                    record.perplexity = metrics.get("perplexity")
                    record.per_stock_accuracy = metrics.get("per_stock_accuracy", {})
                    record.vocab_size = metrics.get("vocab_size")
                    record.num_words = metrics.get("num_words")
                    record.train_sequences = metrics.get("train_sequences")
                    record.eval_sequences = metrics.get("eval_sequences")
                    record.training_seconds = metrics.get("training_seconds")
                    record.metrics_source = "training.log"
            else:
                metrics = parquet_index.lookup(record.tickers, run_timestamp)
                if metrics:
                    record.eval_accuracy = metrics["eval_accuracy"]
                    record.eval_loss = metrics["eval_loss"]
                    record.perplexity = metrics["perplexity"]
                    record.metrics_source = "models.parquet"

            records.append(record)

    return records


def records_to_dataframe(records: List[ModelRecord]) -> pd.DataFrame:
    """Flatten a list of :class:`ModelRecord` into a :class:`pandas.DataFrame`."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([r.to_flat_dict() for r in records])


def top_models_df(df: pd.DataFrame, n: int = 10, metric: str = "eval_accuracy") -> pd.DataFrame:
    """Return the top ``n`` rows of ``df`` ranked by ``metric`` (descending).

    Models with no recorded metric are sorted to the bottom.

    Args:
        df: DataFrame as produced by :func:`records_to_dataframe`.
        n: Number of rows to return (default: 10).
        metric: Column to rank by (default: ``'eval_accuracy'``).

    Returns:
        The top ``n`` rows, sorted best-first.
    """
    if df.empty:
        return df
    sorted_df = df.sort_values(
        by=[metric, "eval_loss"], ascending=[False, True], na_position="last"
    )
    return sorted_df.head(n).reset_index(drop=True)


def search_models_df(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Return all rows whose ticker set is a superset of ``tickers``.

    Matching is case-insensitive.

    Args:
        df: DataFrame as produced by :func:`records_to_dataframe`.
        tickers: Tickers that must all be present in a model for it to match.

    Returns:
        Matching rows, sorted by ``eval_accuracy`` descending.
    """
    if df.empty or not tickers:
        return df.iloc[0:0]
    wanted = {t.strip().upper() for t in tickers if t.strip()}
    mask = df["tickers"].apply(lambda ts: wanted.issubset({t.upper() for t in ts}))
    matched = df[mask]
    return top_models_df(matched, n=len(matched)) if not matched.empty else matched
