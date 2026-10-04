"""Unit tests for backfill_registry.py module."""
from datetime import datetime, timedelta

import pandas as pd
import pytest
import yaml

from pytink.backfill_registry import add_missing_entries, add_model_type

from tests.test_model_registry import _make_run_dir, _make_parquet, _parquet_row


# ── add_missing_entries ───────────────────────────────────────────────────────

class TestAddMissingEntries:
    """Tests for add_missing_entries()."""

    def test_adds_row_for_run_with_training_log_not_in_parquet(self, tmp_path):
        """A training.log run with no matching parquet row gets a new row added."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        _make_run_dir(
            models_dir, "AAPL-MSFT", "20260104_192439", tickers=["AAPL", "MSFT"],
            with_log=True, eval_loss=0.61, eval_accuracy=0.94, perplexity=1.85,
        )

        summary = add_missing_entries(parquet_path=parquet_path, models_dir=models_dir)

        assert summary["added"] == 1
        df = pd.read_parquet(parquet_path)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["tickers"] == "AAPL-MSFT"
        assert row["accuracy"] == pytest.approx(0.94)
        assert row["loss"] == pytest.approx(0.61)
        assert row["perplexity"] == pytest.approx(1.85)
        assert row["model_type"] == "gpt2"
        assert pd.isna(row["quote_start_date"])

    def test_skips_run_already_present_in_parquet(self, tmp_path):
        """A ticker set + close timestamp already in models.parquet is not duplicated."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        run_ts = datetime(2026, 8, 1, 12, 5, 0)
        _make_run_dir(models_dir, "AAPL-MSFT", run_ts.strftime("%Y%m%d_%H%M%S"),
                      tickers=["AAPL", "MSFT"], with_log=True)
        _make_parquet(parquet_path, [
            _parquet_row(["AAPL", "MSFT"], accuracy=0.5, loss=1.0, perplexity=2.0,
                         created_at=run_ts + timedelta(seconds=30)),
        ])

        summary = add_missing_entries(parquet_path=parquet_path, models_dir=models_dir)

        assert summary["added"] == 0
        assert summary["skipped_already_present"] == 1
        df = pd.read_parquet(parquet_path)
        assert len(df) == 1

    def test_farm_trained_runs_without_log_are_not_added(self, tmp_path):
        """Runs without a training.log (farm-trained) are not candidates for addition."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        _make_run_dir(models_dir, "AAPL-MSFT", "20260801_120500",
                      tickers=["AAPL", "MSFT"], with_log=False)

        summary = add_missing_entries(parquet_path=parquet_path, models_dir=models_dir)

        assert summary["added"] == 0
        assert not parquet_path.exists()

    def test_creates_parquet_when_missing(self, tmp_path):
        """A new models.parquet is created when none existed."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        _make_run_dir(models_dir, "AAPL-MSFT", "20260104_192439", tickers=["AAPL", "MSFT"])

        assert not parquet_path.exists()
        add_missing_entries(parquet_path=parquet_path, models_dir=models_dir)
        assert parquet_path.exists()

    def test_dry_run_does_not_write(self, tmp_path):
        """dry_run=True reports the addition but leaves the parquet file untouched."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        _make_run_dir(models_dir, "AAPL-MSFT", "20260104_192439", tickers=["AAPL", "MSFT"])

        summary = add_missing_entries(parquet_path=parquet_path, models_dir=models_dir, dry_run=True)

        assert summary["added"] == 1
        assert not parquet_path.exists()

    def test_uses_custom_model_type(self, tmp_path):
        """A non-default model_type is written into the new row."""
        models_dir = tmp_path / "models"
        parquet_path = tmp_path / "models.parquet"
        _make_run_dir(models_dir, "AAPL-MSFT", "20260104_192439", tickers=["AAPL", "MSFT"])

        add_missing_entries(parquet_path=parquet_path, models_dir=models_dir, model_type="bert")

        df = pd.read_parquet(parquet_path)
        assert df.iloc[0]["model_type"] == "bert"


# ── add_model_type ────────────────────────────────────────────────────────────

class TestAddModelType:
    """Tests for add_model_type()."""

    def test_adds_model_type_to_config_yaml(self, tmp_path):
        """config.yaml gets model_type written into its model: section."""
        models_dir = tmp_path / "models"
        run_dir = _make_run_dir(models_dir, "AAPL-MSFT", "20260104_192439", tickers=["AAPL", "MSFT"])

        summary = add_model_type(parquet_path=tmp_path / "models.parquet", models_dir=models_dir)

        assert summary["configs_updated"] == 1
        with open(run_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["model"]["model_type"] == "gpt2"
        # Original model-section keys are preserved.
        assert config["model"]["hidden_size"] == 128

    def test_skips_config_already_set(self, tmp_path):
        """A config.yaml that already has the target model_type is left alone."""
        models_dir = tmp_path / "models"
        run_dir = _make_run_dir(models_dir, "AAPL-MSFT", "20260104_192439", tickers=["AAPL", "MSFT"])
        with open(run_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
        config["model"]["model_type"] = "gpt2"
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)

        summary = add_model_type(parquet_path=tmp_path / "models.parquet", models_dir=models_dir)

        assert summary["configs_updated"] == 0
        assert summary["configs_already_set"] == 1

    def test_updates_parquet_rows_missing_model_type(self, tmp_path):
        """Existing models.parquet rows without model_type get it filled in."""
        parquet_path = tmp_path / "models.parquet"
        _make_parquet(parquet_path, [
            _parquet_row(["AAPL", "MSFT"], accuracy=0.5, loss=1.0, perplexity=2.0,
                         created_at=datetime(2026, 1, 1)),
        ])
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        summary = add_model_type(parquet_path=parquet_path, models_dir=models_dir)

        assert summary["parquet_rows_updated"] == 1
        df = pd.read_parquet(parquet_path)
        assert df.iloc[0]["model_type"] == "gpt2"

    def test_leaves_existing_parquet_model_type_untouched(self, tmp_path):
        """A row that already has a model_type is not overwritten."""
        parquet_path = tmp_path / "models.parquet"
        row = _parquet_row(["AAPL", "MSFT"], accuracy=0.5, loss=1.0, perplexity=2.0,
                           created_at=datetime(2026, 1, 1))
        row["model_type"] = "bert"
        _make_parquet(parquet_path, [row])
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        summary = add_model_type(parquet_path=parquet_path, models_dir=models_dir)

        assert summary["parquet_rows_updated"] == 0
        df = pd.read_parquet(parquet_path)
        assert df.iloc[0]["model_type"] == "bert"

    def test_dry_run_does_not_modify_config_or_parquet(self, tmp_path):
        """dry_run=True reports changes but writes nothing."""
        models_dir = tmp_path / "models"
        run_dir = _make_run_dir(models_dir, "AAPL-MSFT", "20260104_192439", tickers=["AAPL", "MSFT"])
        parquet_path = tmp_path / "models.parquet"
        _make_parquet(parquet_path, [
            _parquet_row(["AAPL", "MSFT"], accuracy=0.5, loss=1.0, perplexity=2.0,
                         created_at=datetime(2026, 1, 1)),
        ])

        summary = add_model_type(parquet_path=parquet_path, models_dir=models_dir, dry_run=True)

        assert summary["configs_updated"] == 1
        assert summary["parquet_rows_updated"] == 1
        with open(run_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
        assert "model_type" not in config["model"]
        df = pd.read_parquet(parquet_path)
        assert "model_type" not in df.columns

    def test_writes_config_backup(self, tmp_path):
        """A .bak copy of config.yaml is created before overwriting."""
        models_dir = tmp_path / "models"
        run_dir = _make_run_dir(models_dir, "AAPL-MSFT", "20260104_192439", tickers=["AAPL", "MSFT"])

        add_model_type(parquet_path=tmp_path / "models.parquet", models_dir=models_dir)

        assert (run_dir / "config.yaml.bak").exists()

    def test_handles_missing_models_dir(self, tmp_path):
        """A nonexistent models_dir is handled without error."""
        summary = add_model_type(
            parquet_path=tmp_path / "models.parquet", models_dir=tmp_path / "nope",
        )
        assert summary["configs_updated"] == 0
