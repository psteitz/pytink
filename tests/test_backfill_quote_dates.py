"""Unit tests for backfill_quote_dates.py module."""
from datetime import datetime, timedelta

import pandas as pd
import pytest
import yaml

from pytink.backfill_quote_dates import backfill


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_model_dir(models_dir, tickers, run_ts, start_date=None, end_date=None):
    """Create models/<sorted-tickers>/<timestamp>/config.yaml."""
    ticker_key = "-".join(sorted(tickers))
    run_dir = models_dir / ticker_key / run_ts.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    data = {"tickers": sorted(tickers), "interval_minutes": 30}
    if start_date:
        data["start_date"] = start_date
    if end_date:
        data["end_date"] = end_date
    config = {"data": data}
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    return run_dir


def _write_parquet(parquet_path, rows):
    """Write a models.parquet file from a list of row dicts."""
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)


# ── backfill() ────────────────────────────────────────────────────────────────

class TestBackfill:
    """Tests for backfill()."""

    def test_missing_parquet_returns_zeroed_summary(self, tmp_path):
        """A nonexistent parquet path yields a zeroed summary, no crash."""
        summary = backfill(
            parquet_path=tmp_path / "nope.parquet",
            models_dir=tmp_path / "models",
        )
        assert summary["total_rows"] == 0
        assert summary["backfilled"] == 0

    def test_backfills_when_dates_present_in_config(self, tmp_path):
        """A row is backfilled when its matching config.yaml has start/end dates."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        run_ts = datetime(2026, 1, 1, 12, 0, 0)

        _write_model_dir(
            models_dir, ["AAPL", "GOOG"], run_ts,
            start_date="2024-01-01", end_date="2024-06-30",
        )
        _write_parquet(parquet_path, [
            {"tickers": "AAPL-GOOG", "created_at": run_ts + timedelta(seconds=5)},
        ])

        summary = backfill(parquet_path=parquet_path, models_dir=models_dir)

        assert summary["backfilled"] == 1
        df = pd.read_parquet(parquet_path)
        assert pd.Timestamp(df.iloc[0]["quote_start_date"]) == pd.Timestamp("2024-01-01")
        assert pd.Timestamp(df.iloc[0]["quote_end_date"]) == pd.Timestamp("2024-06-30")

    def test_reports_dir_found_no_dates_in_config(self, tmp_path):
        """A matching directory without start/end dates is reported, not guessed."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        run_ts = datetime(2026, 1, 1, 12, 0, 0)

        _write_model_dir(models_dir, ["AAPL", "GOOG"], run_ts)
        _write_parquet(parquet_path, [
            {"tickers": "AAPL-GOOG", "created_at": run_ts + timedelta(seconds=5)},
        ])

        summary = backfill(parquet_path=parquet_path, models_dir=models_dir)

        assert summary["dir_found_no_dates_in_config"] == 1
        assert summary["backfilled"] == 0
        # Nothing was backfilled, so the file is left untouched (no new column).
        df = pd.read_parquet(parquet_path)
        assert "quote_start_date" not in df.columns

    def test_reports_no_matching_dir(self, tmp_path):
        """A row whose ticker set has no model directory is reported as such."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        parquet_path = tmp_path / "models.parquet"
        _write_parquet(parquet_path, [
            {"tickers": "ZZZZ-YYYY", "created_at": datetime(2026, 1, 1)},
        ])

        summary = backfill(parquet_path=parquet_path, models_dir=models_dir)

        assert summary["no_matching_dir"] == 1
        assert summary["backfilled"] == 0

    def test_reports_no_dir_within_tolerance(self, tmp_path):
        """A matching ticker set whose directory timestamp is too far away is skipped."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        run_ts = datetime(2026, 1, 1, 12, 0, 0)

        _write_model_dir(
            models_dir, ["AAPL", "GOOG"], run_ts,
            start_date="2024-01-01", end_date="2024-06-30",
        )
        _write_parquet(parquet_path, [
            {"tickers": "AAPL-GOOG", "created_at": run_ts + timedelta(hours=5)},
        ])

        summary = backfill(parquet_path=parquet_path, models_dir=models_dir, tolerance_seconds=60)

        assert summary["no_dir_within_tolerance"] == 1
        assert summary["backfilled"] == 0

    def test_skips_rows_already_populated(self, tmp_path):
        """Rows that already have both quote dates are left untouched and counted."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        parquet_path = tmp_path / "models.parquet"
        _write_parquet(parquet_path, [
            {
                "tickers": "AAPL-GOOG",
                "created_at": datetime(2026, 1, 1),
                "quote_start_date": datetime(2024, 1, 1),
                "quote_end_date": datetime(2024, 6, 30),
            },
        ])

        summary = backfill(parquet_path=parquet_path, models_dir=models_dir)

        assert summary["already_present"] == 1
        assert summary["backfilled"] == 0

    def test_dry_run_does_not_modify_file(self, tmp_path):
        """dry_run=True computes results but leaves the parquet file unchanged."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        run_ts = datetime(2026, 1, 1, 12, 0, 0)

        _write_model_dir(
            models_dir, ["AAPL", "GOOG"], run_ts,
            start_date="2024-01-01", end_date="2024-06-30",
        )
        _write_parquet(parquet_path, [
            {"tickers": "AAPL-GOOG", "created_at": run_ts + timedelta(seconds=5)},
        ])

        summary = backfill(parquet_path=parquet_path, models_dir=models_dir, dry_run=True)

        assert summary["backfilled"] == 1
        df = pd.read_parquet(parquet_path)
        assert "quote_start_date" not in df.columns or pd.isna(df.iloc[0]["quote_start_date"])

    def test_writes_backup_file_when_backfilling(self, tmp_path):
        """A .bak copy of the original parquet is created before overwriting."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        run_ts = datetime(2026, 1, 1, 12, 0, 0)

        _write_model_dir(
            models_dir, ["AAPL", "GOOG"], run_ts,
            start_date="2024-01-01", end_date="2024-06-30",
        )
        _write_parquet(parquet_path, [
            {"tickers": "AAPL-GOOG", "created_at": run_ts + timedelta(seconds=5)},
        ])

        backfill(parquet_path=parquet_path, models_dir=models_dir)

        assert (tmp_path / "models.parquet.bak").exists()

    def test_picks_closest_run_when_multiple_candidates(self, tmp_path):
        """When multiple runs share a ticker set, the closest-in-time one is used."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        far_ts = datetime(2026, 1, 1, 0, 0, 0)
        near_ts = datetime(2026, 1, 1, 12, 0, 0)

        _write_model_dir(
            models_dir, ["AAPL", "GOOG"], far_ts,
            start_date="2020-01-01", end_date="2020-06-30",
        )
        _write_model_dir(
            models_dir, ["AAPL", "GOOG"], near_ts,
            start_date="2024-01-01", end_date="2024-06-30",
        )
        _write_parquet(parquet_path, [
            {"tickers": "AAPL-GOOG", "created_at": near_ts + timedelta(seconds=5)},
        ])

        summary = backfill(parquet_path=parquet_path, models_dir=models_dir)

        assert summary["backfilled"] == 1
        df = pd.read_parquet(parquet_path)
        assert pd.Timestamp(df.iloc[0]["quote_start_date"]) == pd.Timestamp("2024-01-01")
