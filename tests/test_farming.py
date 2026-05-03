"""Unit tests for farming.py module."""
import tempfile
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch, call

from pytink.farming import (
    ModelEntry,
    ModelFarm,
    DEFAULT_MODELS_DIR,
    HIDDEN_SIZE,
    NUM_HIDDEN_LAYERS,
    NUM_ATTENTION_HEADS,
    MAX_POSITION_EMBEDDINGS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_entry(
    tickers=None,
    eval_loss=1.0,
    eval_accuracy=0.5,
    generation=0,
    num_words=500,
):
    """Create a ModelEntry with a mock model."""
    return ModelEntry(
        model=MagicMock(),
        tickers=tickers or ["AAPL", "GOOG"],
        stock_ids=[1, 2],
        vocab={"abc": 0, "def": 1},
        eval_loss=eval_loss,
        eval_accuracy=eval_accuracy,
        num_words=num_words,
        generation=generation,
    )


@pytest.fixture
def farm(tmp_path):
    """Return a ModelFarm with a mocked database and parquet path."""
    with patch("pytink.farming.StockDatabase") as MockDB, \
         patch("pytink.farming.MODELS_PARQUET_PATH", tmp_path / "models.parquet"):
        mock_db_instance = MagicMock()
        MockDB.return_value = mock_db_instance
        f = ModelFarm(
            db_password="test",
            num_models=4,
            num_generations=2,
        )
        # Expose tmp parquet path on the fixture for assertions
        f._test_parquet_path = tmp_path / "models.parquet"
        yield f


# ── ModelEntry ────────────────────────────────────────────────────────────────

class TestModelEntry:
    """Tests for ModelEntry dataclass-like class."""

    def test_initialization_stores_fields(self):
        """All constructor arguments are stored correctly."""
        entry = _make_entry(tickers=["AAPL", "TSLA"], eval_loss=0.8, eval_accuracy=0.6)
        assert entry.tickers == ["AAPL", "TSLA"]
        assert entry.eval_loss == pytest.approx(0.8)
        assert entry.eval_accuracy == pytest.approx(0.6)
        assert entry.num_words == 500
        assert entry.generation == 0

    def test_perplexity_computed_from_loss(self):
        """Perplexity equals exp(eval_loss) for normal loss values."""
        loss = 1.5
        entry = _make_entry(eval_loss=loss)
        assert entry.perplexity == pytest.approx(float(np.exp(loss)), rel=1e-5)

    def test_perplexity_capped_at_large_loss(self):
        """Perplexity is capped (loss clamped to 88) to avoid overflow."""
        entry = _make_entry(eval_loss=200.0)
        assert entry.perplexity == pytest.approx(float(np.exp(88)), rel=1e-5)

    def test_perplexity_zero_loss(self):
        """Zero loss yields perplexity of 1."""
        entry = _make_entry(eval_loss=0.0)
        assert entry.perplexity == pytest.approx(1.0, rel=1e-5)

    def test_created_at_is_datetime(self):
        """created_at is set to a datetime on construction."""
        before = datetime.now()
        entry = _make_entry()
        after = datetime.now()
        assert before <= entry.created_at <= after

    def test_repr_contains_key_fields(self):
        """__repr__ includes tickers, accuracy, loss, and perplexity."""
        entry = _make_entry(tickers=["AAPL"], eval_loss=1.2, eval_accuracy=0.55)
        r = repr(entry)
        assert "AAPL" in r
        assert "0.5500" in r
        assert "1.2000" in r
        assert "perplexity" in r


# ── ModelFarm._sort_pool ──────────────────────────────────────────────────────

class TestSortPool:
    """Tests for ModelFarm._sort_pool."""

    def test_sort_by_accuracy_descending(self, farm):
        """Highest accuracy model comes first."""
        farm.models = [
            _make_entry(eval_accuracy=0.4, eval_loss=1.0),
            _make_entry(eval_accuracy=0.7, eval_loss=1.0),
            _make_entry(eval_accuracy=0.5, eval_loss=1.0),
        ]
        farm._sort_pool()
        assert farm.models[0].eval_accuracy == pytest.approx(0.7)
        assert farm.models[-1].eval_accuracy == pytest.approx(0.4)

    def test_sort_tiebreak_by_loss_ascending(self, farm):
        """When accuracy is equal, lower loss ranks higher."""
        farm.models = [
            _make_entry(eval_accuracy=0.6, eval_loss=2.0),
            _make_entry(eval_accuracy=0.6, eval_loss=0.5),
            _make_entry(eval_accuracy=0.6, eval_loss=1.0),
        ]
        farm._sort_pool()
        assert farm.models[0].eval_loss == pytest.approx(0.5)
        assert farm.models[-1].eval_loss == pytest.approx(2.0)

    def test_sort_empty_pool(self, farm):
        """Sorting an empty pool does not raise."""
        farm.models = []
        farm._sort_pool()
        assert farm.models == []

    def test_sort_single_model(self, farm):
        """Sorting a one-model pool keeps that model."""
        entry = _make_entry()
        farm.models = [entry]
        farm._sort_pool()
        assert farm.models == [entry]


# ── ModelFarm._append_to_parquet ─────────────────────────────────────────────

class TestAppendToParquet:
    """Tests for ModelFarm._append_to_parquet."""

    def test_creates_parquet_when_missing(self, farm, tmp_path):
        """A new parquet file is created when none exists."""
        parquet_path = tmp_path / "models.parquet"
        assert not parquet_path.exists()

        entry = _make_entry(tickers=["AAPL", "GOOG"], eval_loss=1.2, eval_accuracy=0.55)
        with patch("pytink.farming.MODELS_PARQUET_PATH", parquet_path):
            farm._append_to_parquet(entry)

        assert parquet_path.exists()

    def test_written_row_has_correct_columns(self, farm, tmp_path):
        """All expected columns are present in the written row."""
        parquet_path = tmp_path / "models.parquet"
        entry = _make_entry(tickers=["AAPL", "GOOG"], eval_loss=1.2, eval_accuracy=0.55)
        with patch("pytink.farming.MODELS_PARQUET_PATH", parquet_path):
            farm._append_to_parquet(entry)

        df = pd.read_parquet(parquet_path)
        expected_columns = {
            "tickers", "accuracy", "loss", "perplexity",
            "interval_minutes", "context_window_size", "batch_size",
            "epochs", "learning_rate", "weight_decay",
            "early_stopping_patience", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "max_position_embeddings", "created_at",
        }
        assert expected_columns.issubset(set(df.columns))

    def test_written_row_values_match_entry(self, farm, tmp_path):
        """Values in the parquet row match the ModelEntry and farm parameters."""
        parquet_path = tmp_path / "models.parquet"
        entry = _make_entry(tickers=["AAPL", "GOOG"], eval_loss=1.2, eval_accuracy=0.55)
        with patch("pytink.farming.MODELS_PARQUET_PATH", parquet_path):
            farm._append_to_parquet(entry)

        df = pd.read_parquet(parquet_path)
        row = df.iloc[0]
        assert row["tickers"] == "AAPL-GOOG"
        assert row["accuracy"] == pytest.approx(0.55)
        assert row["loss"] == pytest.approx(1.2)
        assert row["perplexity"] == pytest.approx(entry.perplexity, rel=1e-5)
        assert row["interval_minutes"] == farm.interval_minutes
        assert row["context_window_size"] == farm.context_window_size
        assert row["batch_size"] == farm.batch_size
        assert row["epochs"] == farm.epochs
        assert row["learning_rate"] == pytest.approx(farm.learning_rate)
        assert row["weight_decay"] == pytest.approx(farm.weight_decay)
        assert row["early_stopping_patience"] == farm.early_stopping_patience
        assert row["hidden_size"] == HIDDEN_SIZE
        assert row["num_hidden_layers"] == NUM_HIDDEN_LAYERS
        assert row["num_attention_heads"] == NUM_ATTENTION_HEADS
        assert row["max_position_embeddings"] == MAX_POSITION_EMBEDDINGS

    def test_appends_to_existing_parquet(self, farm, tmp_path):
        """Rows are appended to an existing parquet file, not overwritten."""
        parquet_path = tmp_path / "models.parquet"
        entry1 = _make_entry(tickers=["AAPL"], eval_accuracy=0.5)
        entry2 = _make_entry(tickers=["TSLA"], eval_accuracy=0.6)

        with patch("pytink.farming.MODELS_PARQUET_PATH", parquet_path):
            farm._append_to_parquet(entry1)
            farm._append_to_parquet(entry2)

        df = pd.read_parquet(parquet_path)
        assert len(df) == 2
        assert df.iloc[0]["tickers"] == "AAPL"
        assert df.iloc[1]["tickers"] == "TSLA"

    def test_multiple_appends_grow_row_count(self, farm, tmp_path):
        """Row count grows by one for each call."""
        parquet_path = tmp_path / "models.parquet"
        entries = [_make_entry(tickers=[f"T{i}"], eval_accuracy=i / 10) for i in range(5)]

        with patch("pytink.farming.MODELS_PARQUET_PATH", parquet_path):
            for i, entry in enumerate(entries, start=1):
                farm._append_to_parquet(entry)
                df = pd.read_parquet(parquet_path)
                assert len(df) == i


# ── ModelFarm.display_top_models ──────────────────────────────────────────────

class TestDisplayTopModels:
    """Tests for ModelFarm.display_top_models."""

    def test_prints_header(self, farm, capsys):
        """Output includes the TOP N MODELS header."""
        farm.models = [_make_entry(tickers=["AAPL"])]
        farm.display_top_models(n=1)
        captured = capsys.readouterr().out
        assert "TOP" in captured
        assert "MODELS" in captured

    def test_prints_each_ticker(self, farm, capsys):
        """Each model's tickers appear in the output."""
        farm.models = [
            _make_entry(tickers=["AAPL", "GOOG"]),
            _make_entry(tickers=["TSLA", "MSFT"]),
        ]
        farm.display_top_models(n=2)
        captured = capsys.readouterr().out
        assert "AAPL-GOOG" in captured
        assert "TSLA-MSFT" in captured

    def test_respects_n_limit(self, farm, capsys):
        """Only the top N models are printed."""
        farm.models = [_make_entry(tickers=[f"T{i}"]) for i in range(5)]
        farm.display_top_models(n=2)
        captured = capsys.readouterr().out
        assert "T0" in captured
        assert "T1" in captured
        assert "T2" not in captured

    def test_empty_pool_does_not_raise(self, farm, capsys):
        """Calling display with an empty pool prints an empty table without error."""
        farm.models = []
        farm.display_top_models(n=5)
        captured = capsys.readouterr().out
        assert "TOP" in captured


# ── ModelFarm.cold_start ──────────────────────────────────────────────────────

class TestColdStart:
    """Tests for ModelFarm.cold_start."""

    def test_populates_pool_with_successful_entries(self, farm, tmp_path):
        """cold_start fills the pool with all successful model entries."""
        entry = _make_entry()
        with patch.object(farm, "_build_and_evaluate", return_value=entry), \
             patch.object(farm, "_append_to_parquet") as mock_append, \
             patch("pytink.farming.MODELS_PARQUET_PATH", tmp_path / "models.parquet"):
            farm.cold_start()

        assert len(farm.models) == farm.num_models
        assert mock_append.call_count == farm.num_models

    def test_skips_none_returns(self, farm, tmp_path):
        """cold_start skips None returns from _build_and_evaluate."""
        # Alternate None / valid entry
        entry = _make_entry()
        side_effects = [None if i % 2 == 0 else entry for i in range(farm.num_models)]
        with patch.object(farm, "_build_and_evaluate", side_effect=side_effects), \
             patch.object(farm, "_append_to_parquet") as mock_append, \
             patch("pytink.farming.MODELS_PARQUET_PATH", tmp_path / "models.parquet"):
            farm.cold_start()

        expected_count = sum(1 for e in side_effects if e is not None)
        assert len(farm.models) == expected_count
        assert mock_append.call_count == expected_count

    def test_pool_is_sorted_after_cold_start(self, farm, tmp_path):
        """The pool is sorted by accuracy (descending) after cold_start."""
        entries = [
            _make_entry(eval_accuracy=0.3),
            _make_entry(eval_accuracy=0.7),
            _make_entry(eval_accuracy=0.5),
            _make_entry(eval_accuracy=0.9),
        ]
        with patch.object(farm, "_build_and_evaluate", side_effect=entries), \
             patch.object(farm, "_append_to_parquet"), \
             patch("pytink.farming.MODELS_PARQUET_PATH", tmp_path / "models.parquet"):
            farm.cold_start()

        accuracies = [m.eval_accuracy for m in farm.models]
        assert accuracies == sorted(accuracies, reverse=True)

    def test_generation_counter_stays_zero(self, farm, tmp_path):
        """cold_start passes generation=0 and does not advance self.generation."""
        entry = _make_entry()
        with patch.object(farm, "_build_and_evaluate", return_value=entry), \
             patch.object(farm, "_append_to_parquet"), \
             patch("pytink.farming.MODELS_PARQUET_PATH", tmp_path / "models.parquet"):
            farm.cold_start()

        assert farm.generation == 0


# ── ModelFarm._run_generation ─────────────────────────────────────────────────

class TestRunGeneration:
    """Tests for ModelFarm._run_generation."""

    def _seed_farm(self, farm, count=4):
        """Seed the farm pool with `count` entries of varying accuracy."""
        farm.models = [
            _make_entry(eval_accuracy=(i + 1) / 10) for i in range(count)
        ]
        farm._sort_pool()

    def test_increments_generation_counter(self, farm, tmp_path):
        """Each call to _run_generation increments self.generation by 1."""
        self._seed_farm(farm)
        entry = _make_entry()
        with patch.object(farm, "_build_and_evaluate", return_value=entry), \
             patch.object(farm, "_append_to_parquet"), \
             patch("pytink.farming.MODELS_PARQUET_PATH", tmp_path / "models.parquet"):
            farm._run_generation()
            assert farm.generation == 1
            farm._run_generation()
            assert farm.generation == 2

    def test_keeps_top_quarter_of_pool(self, farm, tmp_path):
        """After _run_generation, the top 25 % of old models are retained."""
        self._seed_farm(farm, count=4)
        top_entry = farm.models[0]  # Highest accuracy
        entry = _make_entry(eval_accuracy=0.0)  # Low accuracy new entries
        with patch.object(farm, "_build_and_evaluate", return_value=entry), \
             patch.object(farm, "_append_to_parquet"), \
             patch("pytink.farming.MODELS_PARQUET_PATH", tmp_path / "models.parquet"):
            farm._run_generation()

        # The best model from before must still be present
        assert top_entry in farm.models

    def test_pool_size_restored_to_num_models(self, farm, tmp_path):
        """After _run_generation, the pool size equals num_models (assuming no None)."""
        self._seed_farm(farm, count=4)
        entry = _make_entry()
        with patch.object(farm, "_build_and_evaluate", return_value=entry), \
             patch.object(farm, "_append_to_parquet"), \
             patch("pytink.farming.MODELS_PARQUET_PATH", tmp_path / "models.parquet"):
            farm._run_generation()

        assert len(farm.models) == farm.num_models

    def test_new_entries_appended_to_parquet(self, farm, tmp_path):
        """_append_to_parquet is called for every new model created in the generation."""
        self._seed_farm(farm, count=4)
        entry = _make_entry()
        keep_count = max(1, len(farm.models) // 4)
        expected_new = farm.num_models - keep_count

        with patch.object(farm, "_build_and_evaluate", return_value=entry), \
             patch.object(farm, "_append_to_parquet") as mock_append, \
             patch("pytink.farming.MODELS_PARQUET_PATH", tmp_path / "models.parquet"):
            farm._run_generation()

        assert mock_append.call_count == expected_new

    def test_pool_sorted_after_generation(self, farm, tmp_path):
        """Pool is sorted by accuracy descending after _run_generation."""
        self._seed_farm(farm, count=4)
        new_entries = [_make_entry(eval_accuracy=i / 10) for i in range(3)]
        with patch.object(farm, "_build_and_evaluate", side_effect=new_entries * 10), \
             patch.object(farm, "_append_to_parquet"), \
             patch("pytink.farming.MODELS_PARQUET_PATH", tmp_path / "models.parquet"):
            farm._run_generation()

        accuracies = [m.eval_accuracy for m in farm.models]
        assert accuracies == sorted(accuracies, reverse=True)


# ── ModelFarm.__init__ (new save_models params) ───────────────────────────────

class TestModelFarmInit:
    """Tests for the save_models and models_dir constructor parameters."""

    def _make_farm(self, **kwargs):
        with patch("pytink.farming.StockDatabase") as MockDB:
            MockDB.return_value = MagicMock()
            return ModelFarm(db_password="test", num_models=2, num_generations=1, **kwargs)

    def test_save_models_defaults_to_true(self):
        """save_models is True when not supplied."""
        farm = self._make_farm()
        assert farm.save_models is True

    def test_save_models_false_stored(self):
        """save_models=False is stored on the instance."""
        farm = self._make_farm(save_models=False)
        assert farm.save_models is False

    def test_models_dir_defaults_to_default_models_dir(self):
        """models_dir falls back to DEFAULT_MODELS_DIR when not supplied."""
        farm = self._make_farm()
        assert farm.models_dir == DEFAULT_MODELS_DIR

    def test_models_dir_custom_path_stored(self, tmp_path):
        """A custom models_dir is stored unchanged."""
        farm = self._make_farm(models_dir=tmp_path)
        assert farm.models_dir == tmp_path


# ── ModelFarm._build_and_evaluate — save_model integration ────────────────────

class TestBuildAndEvaluateSaveModels:
    """Tests that _build_and_evaluate calls save_model iff save_models is True."""

    _ELIGIBLE = [
        {"id": i, "ticker": t}
        for i, t in enumerate(["AAPL", "GOOG", "MSFT", "TSLA", "AMZN"], start=1)
    ]
    _QUOTES = {
        i: [{"timestamp": "2026-01-01T00:00:00", "close": 100.0}]
        for i in range(1, 6)
    }

    def _run_build(self, farm, save_models_val, tmp_path):
        """Run _build_and_evaluate with standard mocks; return (result, mock_save)."""
        farm.save_models = save_models_val
        farm.models_dir = tmp_path
        farm.min_stocks = 2
        farm.max_stocks = 2

        prepared = MagicMock()
        prepared.words = ["w"] * 100
        prepared.vocab = {"w": 0}
        prepared.train_loader = MagicMock()
        prepared.eval_loader = MagicMock()
        prepared.eval_subset = [MagicMock()] * 20

        with patch.object(farm, "_get_eligible_stocks", return_value=self._ELIGIBLE), \
             patch.object(farm.db, "get_quotes_for_stocks", return_value=self._QUOTES), \
             patch("pytink.farming.prepare_data", return_value=prepared), \
             patch("pytink.farming.StockTransformerModel"), \
             patch("pytink.farming.train_and_evaluate", return_value=(0.5, 0.7)), \
             patch("pytink.farming.save_model") as mock_save:
            result = farm._build_and_evaluate(generation=0)
            return result, mock_save

    def test_save_model_called_when_save_models_true(self, farm, tmp_path):
        """save_model is called exactly once when save_models=True."""
        result, mock_save = self._run_build(farm, save_models_val=True, tmp_path=tmp_path)
        assert result is not None
        mock_save.assert_called_once()

    def test_save_model_not_called_when_save_models_false(self, farm, tmp_path):
        """save_model is never called when save_models=False."""
        result, mock_save = self._run_build(farm, save_models_val=False, tmp_path=tmp_path)
        assert result is not None
        mock_save.assert_not_called()

    def test_save_model_receives_models_dir(self, farm, tmp_path):
        """The second positional argument to save_model is models_dir."""
        result, mock_save = self._run_build(farm, save_models_val=True, tmp_path=tmp_path)
        assert result is not None
        pos_args, _ = mock_save.call_args
        assert pos_args[1] == tmp_path

    def test_save_model_receives_tickers_kwarg(self, farm, tmp_path):
        """save_model is called with a tickers keyword argument."""
        result, mock_save = self._run_build(farm, save_models_val=True, tmp_path=tmp_path)
        assert result is not None
        _, kw = mock_save.call_args
        assert "tickers" in kw
        assert all(t in ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN"] for t in kw["tickers"])

    def test_save_model_receives_args_kwarg(self, farm, tmp_path):
        """save_model is called with an args keyword argument containing training params."""
        result, mock_save = self._run_build(farm, save_models_val=True, tmp_path=tmp_path)
        assert result is not None
        _, kw = mock_save.call_args
        assert "args" in kw
        farm_args = kw["args"]
        assert farm_args.epochs == farm.epochs
        assert farm_args.batch_size == farm.batch_size
        assert farm_args.learning_rate == pytest.approx(farm.learning_rate)
