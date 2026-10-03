"""Unit tests for model_registry.py module."""
import pandas as pd
import pytest
import yaml

from pytink.model_registry import (
    ModelRecord,
    _parse_training_log,
    records_to_dataframe,
    scan_models_dir,
    search_models_df,
    top_models_df,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_config(run_dir, tickers, interval_minutes=30, context_window_size=128):
    """Write a minimal config.yaml into run_dir."""
    config = {
        "data": {
            "interval_minutes": interval_minutes,
            "context_window_size": context_window_size,
            "tickers": sorted(tickers),
        },
        "model": {
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "max_position_embeddings": 256,
        },
        "training": {
            "batch_size": 64,
            "num_epochs": 5,
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "early_stopping_patience": 3,
        },
        "output": {"save_model": True},
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    (run_dir / "model.pt").write_text("fake-weights")
    return config


def _training_log_text(eval_loss=0.5, eval_accuracy=0.9, perplexity=1.6):
    return (
        "2026-01-04 18:48:03,044 - __main__ - INFO - Starting training for 5 epochs...\n"
        "2026-01-04 18:48:03,044 - __main__ - INFO - Unique words: 1837\n"
        "2026-01-04 18:48:03,044 - __main__ - INFO - Generated 408062 words\n"
        "2026-01-04 18:48:03,044 - __main__ - INFO - Train sequences: 346635, Eval sequences: 61171\n"
        f"2026-01-04 19:24:11,948 - __main__ - INFO - Model training completed in 2173.39 seconds (36.22 minutes)\n"
        f"2026-01-04 19:24:25,848 - __main__ - INFO - Final Eval Loss: {eval_loss}\n"
        f"2026-01-04 19:24:25,848 - __main__ - INFO - Final Eval Accuracy: {eval_accuracy}\n"
        f"2026-01-04 19:24:25,848 - __main__ - INFO - Final Perplexity: {perplexity}\n"
        "2026-01-04 19:24:39,763 - __main__ - INFO - \nVTI (position 1/5) - Accuracy: 0.9941\n"
        "2026-01-04 19:24:39,780 - __main__ - INFO - \nTSN (position 2/5) - Accuracy: 0.9854\n"
    )


def _make_run_dir(models_dir, ticker_key, timestamp, tickers=None, with_log=True, **log_kwargs):
    """Create models_dir/<ticker_key>/<timestamp>/ with config.yaml (+training.log)."""
    run_dir = models_dir / ticker_key / timestamp
    run_dir.mkdir(parents=True)
    _write_config(run_dir, tickers or ticker_key.split("-"))
    if with_log:
        (run_dir / "training.log").write_text(_training_log_text(**log_kwargs))
    return run_dir


def _make_parquet(path, rows):
    """Write a models.parquet file from a list of row dicts."""
    pd.DataFrame(rows).to_parquet(path, index=False)


def _parquet_row(tickers, accuracy, loss, perplexity, created_at):
    return {
        "tickers": "-".join(tickers),
        "accuracy": accuracy,
        "loss": loss,
        "perplexity": perplexity,
        "interval_minutes": 30,
        "context_window_size": 128,
        "batch_size": 64,
        "epochs": 5,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "early_stopping_patience": 3,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "max_position_embeddings": 256,
        "created_at": pd.Timestamp(created_at),
    }


# ── _parse_training_log ──────────────────────────────────────────────────────

class TestParseTrainingLog:
    """Tests for _parse_training_log."""

    def test_extracts_final_metrics(self, tmp_path):
        """Final eval loss/accuracy/perplexity are extracted."""
        log_path = tmp_path / "training.log"
        log_path.write_text(_training_log_text(eval_loss=0.61, eval_accuracy=0.93, perplexity=1.84))
        result = _parse_training_log(log_path)
        assert result["eval_loss"] == pytest.approx(0.61)
        assert result["eval_accuracy"] == pytest.approx(0.93)
        assert result["perplexity"] == pytest.approx(1.84)

    def test_extracts_per_stock_accuracy(self, tmp_path):
        """Per-stock accuracy lines are parsed into a dict."""
        log_path = tmp_path / "training.log"
        log_path.write_text(_training_log_text())
        result = _parse_training_log(log_path)
        assert result["per_stock_accuracy"] == {"VTI": 0.9941, "TSN": 0.9854}

    def test_extracts_word_and_sequence_counts(self, tmp_path):
        """Vocab size, word count, and sequence counts are parsed."""
        log_path = tmp_path / "training.log"
        log_path.write_text(_training_log_text())
        result = _parse_training_log(log_path)
        assert result["vocab_size"] == 1837
        assert result["num_words"] == 408062
        assert result["train_sequences"] == 346635
        assert result["eval_sequences"] == 61171
        assert result["training_seconds"] == pytest.approx(2173.39)

    def test_missing_file_returns_empty_dict(self, tmp_path):
        """A nonexistent log path returns {} rather than raising."""
        result = _parse_training_log(tmp_path / "does_not_exist.log")
        assert result == {}


# ── scan_models_dir ──────────────────────────────────────────────────────────

class TestScanModelsDir:
    """Tests for scan_models_dir."""

    def test_missing_models_dir_returns_empty_list(self, tmp_path):
        """A nonexistent models directory yields no records."""
        records = scan_models_dir(tmp_path / "nope", tmp_path / "models.parquet")
        assert records == []

    def test_skips_ticker_dir_without_dated_subdir(self, tmp_path):
        """Legacy flat layout (files directly under the ticker dir) is ignored."""
        models_dir = tmp_path / "models"
        flat_dir = models_dir / "AAPL-MSFT"
        flat_dir.mkdir(parents=True)
        (flat_dir / "model.pt").write_text("legacy")
        (flat_dir / "config.yaml").write_text("data: {}\n")

        records = scan_models_dir(models_dir, tmp_path / "models.parquet")
        assert records == []

    def test_skips_dated_subdir_without_config(self, tmp_path):
        """A timestamp directory with no config.yaml is ignored."""
        models_dir = tmp_path / "models"
        run_dir = models_dir / "AAPL-MSFT" / "20260101_120000"
        run_dir.mkdir(parents=True)
        (run_dir / "model.pt").write_text("weights-only")

        records = scan_models_dir(models_dir, tmp_path / "models.parquet")
        assert records == []

    def test_parses_run_with_training_log(self, tmp_path):
        """A run with training.log gets metrics from the log, not the parquet."""
        models_dir = tmp_path / "models"
        _make_run_dir(models_dir, "TSN-VTI", "20260104_192439", tickers=["TSN", "VTI"],
                      with_log=True, eval_loss=0.61, eval_accuracy=0.94, perplexity=1.85)

        records = scan_models_dir(models_dir, tmp_path / "models.parquet")
        assert len(records) == 1
        record = records[0]
        assert record.tickers == ["TSN", "VTI"]
        assert record.metrics_source == "training.log"
        assert record.eval_accuracy == pytest.approx(0.94)
        assert record.per_stock_accuracy == {"VTI": 0.9941, "TSN": 0.9854}
        assert record.interval_minutes == 30
        assert record.context_window_size == 128

    def test_falls_back_to_parquet_when_no_training_log(self, tmp_path):
        """A run without training.log pulls metrics from models.parquet by ticker+time."""
        models_dir = tmp_path / "models"
        run_dir = _make_run_dir(models_dir, "AAPL-MSFT", "20260801_120500",
                                 tickers=["AAPL", "MSFT"], with_log=False)

        parquet_path = tmp_path / "models.parquet"
        _make_parquet(parquet_path, [
            _parquet_row(["MSFT", "AAPL"], accuracy=0.77, loss=1.1, perplexity=3.0,
                         created_at="2026-08-01 12:04:58"),
        ])

        records = scan_models_dir(models_dir, parquet_path)
        assert len(records) == 1
        record = records[0]
        assert record.metrics_source == "models.parquet"
        assert record.eval_accuracy == pytest.approx(0.77)
        assert record.eval_loss == pytest.approx(1.1)
        assert record.perplexity == pytest.approx(3.0)
        assert record.model_dir == run_dir

    def test_parquet_lookup_picks_closest_timestamp(self, tmp_path):
        """When multiple parquet rows share a ticker set, the nearest in time wins."""
        models_dir = tmp_path / "models"
        _make_run_dir(models_dir, "AAPL-MSFT", "20260801_120500",
                      tickers=["AAPL", "MSFT"], with_log=False)

        parquet_path = tmp_path / "models.parquet"
        _make_parquet(parquet_path, [
            _parquet_row(["AAPL", "MSFT"], accuracy=0.11, loss=9.9, perplexity=99.0,
                         created_at="2026-07-01 00:00:00"),  # far away
            _parquet_row(["AAPL", "MSFT"], accuracy=0.77, loss=1.1, perplexity=3.0,
                         created_at="2026-08-01 12:04:58"),  # close match
        ])

        records = scan_models_dir(models_dir, parquet_path)
        assert records[0].eval_accuracy == pytest.approx(0.77)

    def test_no_matching_parquet_row_leaves_metrics_none(self, tmp_path):
        """No training.log and no matching parquet row leaves metrics unset."""
        models_dir = tmp_path / "models"
        _make_run_dir(models_dir, "AAPL-MSFT", "20260801_120500",
                      tickers=["AAPL", "MSFT"], with_log=False)

        records = scan_models_dir(models_dir, tmp_path / "models.parquet")
        record = records[0]
        assert record.metrics_source == "none"
        assert record.eval_accuracy is None
        assert record.eval_loss is None
        assert record.perplexity is None

    def test_config_backward_compat_sequence_length_key(self, tmp_path):
        """Old configs using 'sequence_length' instead of 'context_window_size' still work."""
        models_dir = tmp_path / "models"
        run_dir = models_dir / "AAPL-MSFT" / "20260101_120000"
        run_dir.mkdir(parents=True)
        config = {
            "data": {"interval_minutes": 30, "sequence_length": 256, "tickers": ["AAPL", "MSFT"]},
            "model": {}, "training": {},
        }
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)

        records = scan_models_dir(models_dir, tmp_path / "models.parquet")
        assert records[0].context_window_size == 256


# ── records_to_dataframe / top_models_df / search_models_df ─────────────────

class TestDataFrameHelpers:
    """Tests for records_to_dataframe, top_models_df, and search_models_df."""

    def _records(self):
        return [
            ModelRecord(tickers=["AAPL", "MSFT"], model_dir="d1", run_timestamp=None,
                        eval_accuracy=0.9, eval_loss=0.5),
            ModelRecord(tickers=["GOOG", "MSFT"], model_dir="d2", run_timestamp=None,
                        eval_accuracy=0.7, eval_loss=1.0),
            ModelRecord(tickers=["AAPL", "TSLA"], model_dir="d3", run_timestamp=None,
                        eval_accuracy=None, eval_loss=None),
        ]

    def test_records_to_dataframe_empty_list(self):
        """An empty record list returns an empty DataFrame."""
        df = records_to_dataframe([])
        assert df.empty

    def test_records_to_dataframe_flattens_fields(self):
        """Tickers, metrics, and config params appear as DataFrame columns."""
        df = records_to_dataframe(self._records())
        assert len(df) == 3
        assert set(["tickers_display", "eval_accuracy", "eval_loss"]).issubset(df.columns)
        assert df.iloc[0]["tickers_display"] == "AAPL-MSFT"

    def test_top_models_df_sorts_descending_with_na_last(self):
        """Models rank by eval_accuracy descending; missing metrics sort last."""
        df = records_to_dataframe(self._records())
        top = top_models_df(df, n=3)
        assert list(top["tickers_display"]) == ["AAPL-MSFT", "GOOG-MSFT", "AAPL-TSLA"]

    def test_top_models_df_respects_n(self):
        """Only the requested number of rows are returned."""
        df = records_to_dataframe(self._records())
        top = top_models_df(df, n=1)
        assert len(top) == 1
        assert top.iloc[0]["tickers_display"] == "AAPL-MSFT"

    def test_search_models_df_requires_all_tickers(self):
        """Only models containing every requested ticker match."""
        df = records_to_dataframe(self._records())
        result = search_models_df(df, ["AAPL", "MSFT"])
        assert list(result["tickers_display"]) == ["AAPL-MSFT"]

    def test_search_models_df_single_ticker(self):
        """A singleton ticker list matches any model containing that ticker."""
        df = records_to_dataframe(self._records())
        result = search_models_df(df, ["MSFT"])
        assert set(result["tickers_display"]) == {"AAPL-MSFT", "GOOG-MSFT"}

    def test_search_models_df_is_case_insensitive(self):
        """Lowercase search tickers still match uppercase stored tickers."""
        df = records_to_dataframe(self._records())
        result = search_models_df(df, ["aapl"])
        assert set(result["tickers_display"]) == {"AAPL-MSFT", "AAPL-TSLA"}

    def test_search_models_df_no_match(self):
        """A ticker not present in any model returns an empty result."""
        df = records_to_dataframe(self._records())
        result = search_models_df(df, ["NOPE"])
        assert result.empty

    def test_search_models_df_empty_ticker_list(self):
        """An empty ticker list returns an empty result rather than everything."""
        df = records_to_dataframe(self._records())
        result = search_models_df(df, [])
        assert result.empty
