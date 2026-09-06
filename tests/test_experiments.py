"""Unit tests for experiments.py module."""
from datetime import date, datetime, time
from unittest.mock import MagicMock, patch

import pytest

from pytink.experiments import (
    BUY_TARGET_DOLLARS,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_INTERVAL_MINUTES,
    DEFAULT_NUM_DATES,
    DEFAULT_PREDICT_STEPS,
    DEFAULT_TICKER_SET,
    HIDDEN_SIZE,
    KNOWN_GOOD_MODELS,
    MAX_POSITION_EMBEDDINGS,
    NUM_ATTENTION_HEADS,
    NUM_HIDDEN_LAYERS,
    SIM_DATE_MAX,
    SIM_DATE_MIN,
    compute_shares,
    get_closing_price,
    is_trading_day,
    make_buy_decisions,
    next_trading_day,
    random_end_date,
    run_buy_hold_experiment,
    select_random_trading_dates,
)
from pytink.processor import PriceProcessor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_prepared(words=None, vocab=None):
    """Return a MagicMock that looks like a PreparedData namedtuple."""
    processor = PriceProcessor()
    if words is None:
        words = ["ggg", "ddd", "ggg"] * 20
    if vocab is None:
        vocab = {"ggg": 0, "ddd": 1}
    prepared = MagicMock()
    prepared.processor = processor
    prepared.words = words
    prepared.vocab = vocab
    prepared.train_loader = MagicMock()
    prepared.eval_loader = MagicMock()
    prepared.train_subset = MagicMock()
    prepared.eval_subset = MagicMock()
    prepared.eval_subset.__len__ = MagicMock(return_value=10)
    return prepared


def _make_training_quotes():
    """Return a minimal quotes dict for two stocks, usable as a prepare_data input."""
    return {
        1: [{"price": 100.0, "timestamp": datetime(2026, 1, 2, 15, 0, 0)}],
        2: [{"price": 50.0, "timestamp": datetime(2026, 1, 2, 15, 0, 0)}],
    }


# ── TestConstants ─────────────────────────────────────────────────────────────

class TestConstants:
    """Verify module-level constants have the expected values."""

    def test_known_good_models_count(self):
        """KNOWN_GOOD_MODELS contains exactly 10 entries."""
        assert len(KNOWN_GOOD_MODELS) == 10

    def test_default_ticker_set_is_first_known_good(self):
        """DEFAULT_TICKER_SET matches the first entry in KNOWN_GOOD_MODELS."""
        assert DEFAULT_TICKER_SET == KNOWN_GOOD_MODELS[0]

    def test_sim_date_range(self):
        """SIM_DATE_MIN starts 2025-06-01 and SIM_DATE_MAX ends 2026-03-31."""
        assert SIM_DATE_MIN == date(2025, 6, 1)
        assert SIM_DATE_MAX == date(2026, 3, 31)
        assert SIM_DATE_MIN < SIM_DATE_MAX

    def test_default_num_dates(self):
        """DEFAULT_NUM_DATES is 3."""
        assert DEFAULT_NUM_DATES == 3

    def test_buy_target_dollars(self):
        """BUY_TARGET_DOLLARS is 5 000."""
        assert BUY_TARGET_DOLLARS == pytest.approx(5_000.0)

    def test_default_predict_steps(self):
        """DEFAULT_PREDICT_STEPS is 3."""
        assert DEFAULT_PREDICT_STEPS == 3

    def test_architecture_constants(self):
        """Model architecture constants match farming.py defaults."""
        assert HIDDEN_SIZE == 128
        assert NUM_HIDDEN_LAYERS == 4
        assert NUM_ATTENTION_HEADS == 4
        assert MAX_POSITION_EMBEDDINGS == 256

    def test_default_interval_and_context(self):
        """Sampling interval and context window size match expected defaults."""
        assert DEFAULT_INTERVAL_MINUTES == 30
        assert DEFAULT_CONTEXT_WINDOW_SIZE == 16
        assert DEFAULT_EPOCHS == 5


# ── TestSelectRandomTradingDates ─────────────────────────────────────────────

class TestSelectRandomTradingDates:
    """Tests for select_random_trading_dates helper."""

    def test_returns_correct_count(self):
        """Returns exactly n dates."""
        dates = select_random_trading_dates(3)
        assert len(dates) == 3

    def test_all_are_trading_days(self):
        """Every returned date satisfies is_trading_day."""
        for d in select_random_trading_dates(5):
            assert is_trading_day(d), f"{d} is not a trading day"

    def test_dates_are_sorted_ascending(self):
        """Returned dates are in ascending order."""
        dates = select_random_trading_dates(5)
        assert dates == sorted(dates)

    def test_dates_are_distinct(self):
        """All returned dates are unique."""
        dates = select_random_trading_dates(5)
        assert len(dates) == len(set(dates))

    def test_custom_range_is_respected(self):
        """All returned dates fall within the supplied custom range."""
        lo = date(2026, 1, 5)
        hi = date(2026, 1, 16)
        for d in select_random_trading_dates(3, lo, hi):
            assert lo <= d <= hi

    def test_raises_when_not_enough_trading_days(self):
        """ValueError when n exceeds the number of trading days in the range."""
        # A 3-day weekend window has at most 1 trading day (Friday)
        with pytest.raises(ValueError):
            select_random_trading_dates(5, date(2026, 1, 9), date(2026, 1, 11))

    def test_single_date(self):
        """Requesting 1 date returns a list of length 1."""
        result = select_random_trading_dates(1)
        assert len(result) == 1
        assert is_trading_day(result[0])


# ── TestIsTradingDay ──────────────────────────────────────────────────────────

class TestIsTradingDay:
    """Tests for is_trading_day helper."""

    def test_weekday_non_holiday_is_trading_day(self):
        """A regular Tuesday is a trading day."""
        # 2026-01-06 is a Tuesday
        assert is_trading_day(date(2026, 1, 6)) is True

    def test_friday_non_holiday_is_trading_day(self):
        """A regular Friday is a trading day."""
        # 2026-01-09 is a Friday
        assert is_trading_day(date(2026, 1, 9)) is True

    def test_saturday_is_not_trading_day(self):
        """Saturdays are never trading days."""
        # 2026-01-10 is a Saturday
        assert is_trading_day(date(2026, 1, 10)) is False

    def test_sunday_is_not_trading_day(self):
        """Sundays are never trading days."""
        # 2026-01-11 is a Sunday
        assert is_trading_day(date(2026, 1, 11)) is False

    def test_new_years_day_is_not_trading_day(self):
        """New Year's Day 2026 (Thursday) is a market holiday, not a trading day."""
        assert is_trading_day(date(2026, 1, 1)) is False

    def test_christmas_2026_is_not_trading_day(self):
        """Christmas 2026 (Friday) is a market holiday, not a trading day."""
        assert is_trading_day(date(2026, 12, 25)) is False


# ── TestNextTradingDay ────────────────────────────────────────────────────────

class TestNextTradingDay:
    """Tests for next_trading_day helper."""

    def test_weekday_advances_by_one_day(self):
        """Next trading day after a Monday is Tuesday."""
        # 2026-01-05 is Monday, 2026-01-06 is Tuesday
        assert next_trading_day(date(2026, 1, 5)) == date(2026, 1, 6)

    def test_friday_skips_to_monday(self):
        """Next trading day after a Friday is Monday."""
        # 2026-01-09 is Friday, 2026-01-12 is Monday
        assert next_trading_day(date(2026, 1, 9)) == date(2026, 1, 12)

    def test_thursday_skips_holiday_to_tuesday(self):
        """Next trading day after a Thursday preceding a holiday is the following Tuesday.

        2026-01-01 (Thursday, New Year's) means 2026-12-31 (Wednesday) next day
        is 2026-01-01 (holiday), skip to 2026-01-02 (Friday).
        """
        # 2025-12-31 is a Wednesday; next trading day is 2026-01-02 (Friday, not holiday)
        assert next_trading_day(date(2025, 12, 31)) == date(2026, 1, 2)

    def test_friday_before_monday_holiday_skips_to_tuesday(self):
        """Next trading day after a Friday when Monday is a holiday is Tuesday."""
        # 2026-01-16 is a Friday; 2026-01-19 is MLK Day (holiday); next is 2026-01-20
        assert next_trading_day(date(2026, 1, 16)) == date(2026, 1, 20)

    def test_result_is_always_trading_day(self):
        """next_trading_day always returns a valid trading day."""
        # Start from the Friday before a weekend; result must satisfy is_trading_day
        result = next_trading_day(date(2026, 1, 9))
        assert is_trading_day(result)


# ── TestRandomEndDate ─────────────────────────────────────────────────────────

class TestRandomEndDate:
    """Tests for random_end_date helper."""

    def test_result_in_default_range(self):
        """random_end_date() stays within [SIM_DATE_MIN, SIM_DATE_MAX]."""
        for _ in range(20):
            d = random_end_date()
            assert SIM_DATE_MIN <= d <= SIM_DATE_MAX

    def test_result_in_custom_range(self):
        """Custom min/max bounds are respected."""
        lo = date(2026, 2, 1)
        hi = date(2026, 2, 5)
        for _ in range(20):
            d = random_end_date(lo, hi)
            assert lo <= d <= hi

    def test_degenerate_range_always_returns_min(self):
        """When min_date == max_date, always returns that date."""
        fixed = date(2026, 2, 15)
        for _ in range(10):
            assert random_end_date(fixed, fixed) == fixed

    def test_returns_date_object(self):
        """Return type is a date, not a datetime."""
        assert isinstance(random_end_date(), date)


# ── TestGetClosingPrice ───────────────────────────────────────────────────────

class TestGetClosingPrice:
    """Tests for get_closing_price helper."""

    _TARGET = date(2026, 1, 5)

    def test_returns_last_price_on_target_date(self):
        """Returns the last price when multiple quotes exist on the target date."""
        quotes = [
            {"price": 100.0, "timestamp": datetime(2026, 1, 5, 10, 0, 0)},
            {"price": 102.5, "timestamp": datetime(2026, 1, 5, 14, 0, 0)},
            {"price": 103.0, "timestamp": datetime(2026, 1, 5, 15, 59, 0)},
        ]
        assert get_closing_price(quotes, self._TARGET) == pytest.approx(103.0)

    def test_returns_none_when_no_quotes_on_date(self):
        """Returns None when there are no quotes on the target date."""
        quotes = [
            {"price": 100.0, "timestamp": datetime(2026, 1, 4, 15, 0, 0)},
        ]
        assert get_closing_price(quotes, self._TARGET) is None

    def test_returns_none_for_empty_list(self):
        """Returns None when the quote list is empty."""
        assert get_closing_price([], self._TARGET) is None

    def test_ignores_quotes_outside_target_date(self):
        """Quotes before and after the target date are not included."""
        quotes = [
            {"price": 99.0, "timestamp": datetime(2026, 1, 4, 15, 0, 0)},
            {"price": 105.0, "timestamp": datetime(2026, 1, 5, 12, 0, 0)},
            {"price": 110.0, "timestamp": datetime(2026, 1, 6, 10, 0, 0)},
        ]
        result = get_closing_price(quotes, self._TARGET)
        assert result == pytest.approx(105.0)

    def test_handles_string_timestamps(self):
        """ISO-format string timestamps are parsed correctly."""
        quotes = [
            {"price": 77.5, "timestamp": "2026-01-05T10:30:00"},
        ]
        assert get_closing_price(quotes, self._TARGET) == pytest.approx(77.5)

    def test_returns_float(self):
        """Return type is float even when price is stored as a string."""
        quotes = [{"price": "123.45", "timestamp": datetime(2026, 1, 5, 10, 0, 0)}]
        result = get_closing_price(quotes, self._TARGET)
        assert isinstance(result, float)
        assert result == pytest.approx(123.45)


# ── TestComputeShares ─────────────────────────────────────────────────────────

class TestComputeShares:
    """Tests for compute_shares position-sizing helper."""

    def test_exact_division(self):
        """Exact division: $5000 / $100 = 50 shares."""
        assert compute_shares(100.0, 5000.0) == 50

    def test_rounds_to_nearest_whole_share_down(self):
        """Rounds down when fractional share is below 0.5."""
        # $5000 / $150 = 33.33 → 33
        assert compute_shares(150.0, 5000.0) == 33

    def test_rounds_to_nearest_whole_share_up(self):
        """Rounds up when fractional share is above 0.5."""
        # $5000 / $180 = 27.78 → 28
        assert compute_shares(180.0, 5000.0) == 28

    def test_minimum_one_share_for_high_price(self):
        """Returns at least 1 share when price exceeds target."""
        # $5000 / $10000 = 0.5 → round(0.5) = 0 (banker's) → max(1, 0) = 1
        assert compute_shares(10_000.0, 5000.0) == 1

    def test_minimum_one_share_for_zero_price(self):
        """Returns 1 share when price is 0 (avoids division by zero)."""
        assert compute_shares(0.0, 5000.0) == 1

    def test_minimum_one_share_for_negative_price(self):
        """Returns 1 share when price is negative."""
        assert compute_shares(-100.0, 5000.0) == 1

    def test_uses_default_target_when_not_supplied(self):
        """Default target is BUY_TARGET_DOLLARS."""
        # $5000 / $200 = 25
        assert compute_shares(200.0) == compute_shares(200.0, BUY_TARGET_DOLLARS)


# ── TestMakeBuyDecisions ──────────────────────────────────────────────────────

class TestMakeBuyDecisions:
    """Tests for make_buy_decisions buy-signal logic.

    Uses a real PriceProcessor with default delta_values
    [-.01, -.005, -.001, 0, +.001, +.005, +.01], so:
        buy_letters  = {'f', 'g'}  (top-2 positive deltas: +0.005 and +0.01)
        veto_letters = {'a', 'b'}

    Words are multi-character strings; character at position i maps to ticker i.
    """

    @pytest.fixture
    def processor(self):
        return PriceProcessor()

    def test_buy_both_tickers_on_max_positive_first_token(self, processor):
        """Both tickers are bought when first token is 'gg' (max positive for each)."""
        result = make_buy_decisions(["gg", "dd"], ["A", "B"], processor)
        assert result == ["A", "B"]

    def test_buy_both_tickers_on_second_highest_delta(self, processor):
        """Both tickers are bought when first token is 'ff' (+0.005 each)."""
        result = make_buy_decisions(["ff", "dd"], ["A", "B"], processor)
        assert result == ["A", "B"]

    def test_buy_first_ticker_on_second_highest_delta(self, processor):
        """Ticker A is bought on 'f' (+0.005) while B is not on 'd'."""
        result = make_buy_decisions(["fd"], ["A", "B"], processor)
        assert result == ["A"]

    def test_buy_second_ticker_on_second_highest_delta(self, processor):
        """Ticker B is bought on 'f' (+0.005) while A is not on 'd'."""
        result = make_buy_decisions(["df"], ["A", "B"], processor)
        assert result == ["B"]

    def test_mixed_f_and_g_both_buy(self, processor):
        """One ticker on 'f', the other on 'g' — both meet the buy criterion."""
        result = make_buy_decisions(["fg"], ["A", "B"], processor)
        assert result == ["A", "B"]

    def test_buy_only_first_ticker(self, processor):
        """Only ticker A is bought when first token is 'gd' ('d' is not a buy letter)."""
        result = make_buy_decisions(["gd"], ["A", "B"], processor)
        assert result == ["A"]

    def test_buy_only_second_ticker(self, processor):
        """Only ticker B is bought when first token is 'dg'."""
        result = make_buy_decisions(["dg"], ["A", "B"], processor)
        assert result == ["B"]

    def test_veto_from_later_token_blocks_buy(self, processor):
        """Ticker A is vetoed when a later token has 'a' (strong negative) for A."""
        # words[0] = 'gg' → both have buy signal
        # words[1] = 'ag' → 'a' at position 0 vetoes A; 'g' at position 1 is not a veto
        result = make_buy_decisions(["gg", "ag"], ["A", "B"], processor)
        assert result == ["B"]

    def test_veto_letter_b_also_blocks_buy(self, processor):
        """Ticker A is vetoed when a later token has 'b' (moderate negative) for A."""
        # 'b' is also a veto letter (delta = -0.005, strictly < -0.001)
        result = make_buy_decisions(["gg", "bg"], ["A", "B"], processor)
        assert result == ["B"]

    def test_letter_c_does_not_veto(self, processor):
        """Letter 'c' (delta = -0.001) does NOT veto because -0.001 is not < -0.001."""
        # 'c' maps to delta=-0.001, which is NOT strictly less than -0.001
        result = make_buy_decisions(["gg", "cg"], ["A", "B"], processor)
        assert result == ["A", "B"]

    def test_no_buy_when_first_token_has_no_buy_signal(self, processor):
        """No tickers bought when first token does not encode a top-2 positive delta."""
        result = make_buy_decisions(["dd"], ["A", "B"], processor)
        assert result == []

    def test_e_letter_does_not_trigger_buy(self, processor):
        """Letter 'e' (delta = +0.001) is NOT a buy signal (only top-2 qualify)."""
        result = make_buy_decisions(["ee"], ["A", "B"], processor)
        assert result == []

    def test_empty_predicted_words_returns_no_buys(self, processor):
        """Empty predictions produce no buy decisions."""
        result = make_buy_decisions([], ["A", "B"], processor)
        assert result == []

    def test_none_first_word_returns_no_buys(self, processor):
        """None as the first predicted word produces no buy decisions."""
        result = make_buy_decisions([None], ["A", "B"], processor)
        assert result == []

    def test_single_ticker(self, processor):
        """Single-ticker model: 'g' or 'f' buys, 'e' or lower does not."""
        assert make_buy_decisions(["g"], ["A"], processor) == ["A"]
        assert make_buy_decisions(["f"], ["A"], processor) == ["A"]
        assert make_buy_decisions(["e"], ["A"], processor) == []
        assert make_buy_decisions(["d"], ["A"], processor) == []


# ── TestRunBuyHoldExperiment ──────────────────────────────────────────────────

class TestRunBuyHoldExperiment:
    """Integration-style tests for run_buy_hold_experiment.

    All external dependencies (database, prepare_data, train_and_evaluate,
    predict_next_tokens, StockTransformerModel) are mocked so no real DB or GPU
    is required.
    """

    TICKERS = ["A", "B"]
    END_DATE = date(2026, 1, 5)    # Monday, not a holiday
    SELL_DATE = date(2026, 1, 6)   # Tuesday (next trading day)
    STOCK_DICTS = [{"ticker": "A", "id": 1}, {"ticker": "B", "id": 2}]

    def _make_db(self, training_quotes=None, day_quotes=None):
        """Return a mock StockDatabase with preset return values."""
        db = MagicMock()
        db.get_stocks_by_tickers.return_value = self.STOCK_DICTS
        if day_quotes is None:
            # Single call: return training_quotes only
            db.get_quotes_for_stocks.return_value = training_quotes or _make_training_quotes()
        else:
            db.get_quotes_for_stocks.side_effect = [
                training_quotes or _make_training_quotes(),
                day_quotes,
            ]
        return db

    def _patch_all(self, predicted_words, prepared=None):
        """Context manager that patches all experiment internals."""
        import contextlib
        return contextlib.ExitStack()  # placeholder — see individual tests

    # -- no-buy path -----------------------------------------------------------

    def test_no_buy_decisions_returns_zero_gain(self):
        """Returns zero gain when model predicts no buy signals."""
        db = self._make_db()

        with patch("pytink.experiments.prepare_data", return_value=_make_prepared()), \
             patch("pytink.experiments.train_and_evaluate", return_value=(0.5, 0.8)), \
             patch("pytink.experiments.predict_next_tokens", return_value=["dd"]), \
             patch("pytink.experiments.StockTransformerModel"):
            result = run_buy_hold_experiment(
                db=db, tickers=self.TICKERS, end_date=self.END_DATE, device="cpu"
            )

        assert result["buy_decisions"] == []
        assert result["purchases"] == []
        assert result["total_invested"] == pytest.approx(0.0)
        assert result["total_gain"] == pytest.approx(0.0)
        assert result["return_pct"] == pytest.approx(0.0)

    def test_no_buy_result_carries_train_metrics(self):
        """Even with no buys, train_loss and train_accuracy are returned."""
        db = self._make_db()

        with patch("pytink.experiments.prepare_data", return_value=_make_prepared()), \
             patch("pytink.experiments.train_and_evaluate", return_value=(1.23, 0.65)), \
             patch("pytink.experiments.predict_next_tokens", return_value=["dd"]), \
             patch("pytink.experiments.StockTransformerModel"):
            result = run_buy_hold_experiment(
                db=db, tickers=self.TICKERS, end_date=self.END_DATE, device="cpu"
            )

        assert result["train_loss"] == pytest.approx(1.23)
        assert result["train_accuracy"] == pytest.approx(0.65)
        assert result["end_date"] == self.END_DATE
        assert result["tickers"] == self.TICKERS

    # -- buy path --------------------------------------------------------------

    def test_single_buy_gain_computed_correctly(self):
        """Correct gain when one ticker is bought at $100 and sold at $105."""
        day_quotes = {
            1: [
                {"price": 100.0, "timestamp": datetime(2026, 1, 5, 15, 0, 0)},
                {"price": 105.0, "timestamp": datetime(2026, 1, 6, 15, 0, 0)},
            ]
        }
        db = self._make_db(day_quotes=day_quotes)

        # 'gd': only ticker A (index 0) gets buy signal
        with patch("pytink.experiments.prepare_data", return_value=_make_prepared()), \
             patch("pytink.experiments.train_and_evaluate", return_value=(0.5, 0.8)), \
             patch("pytink.experiments.predict_next_tokens", return_value=["gd"]), \
             patch("pytink.experiments.StockTransformerModel"):
            result = run_buy_hold_experiment(
                db=db, tickers=self.TICKERS, end_date=self.END_DATE,
                buy_target_dollars=5_000.0, device="cpu",
            )

        # 50 shares @ $100, sold @ $105 → gain = $250
        assert result["buy_decisions"] == ["A"]
        assert len(result["purchases"]) == 1
        p = result["purchases"][0]
        assert p["ticker"] == "A"
        assert p["shares"] == 50
        assert p["buy_price"] == pytest.approx(100.0)
        assert p["sell_price"] == pytest.approx(105.0)
        assert p["gain"] == pytest.approx(250.0)
        assert result["total_invested"] == pytest.approx(5_000.0)
        assert result["total_gain"] == pytest.approx(250.0)
        assert result["return_pct"] == pytest.approx(5.0)

    def test_two_buys_total_gain_is_sum(self):
        """total_gain is the sum of individual gains when two stocks are bought."""
        day_quotes = {
            1: [
                {"price": 100.0, "timestamp": datetime(2026, 1, 5, 15, 0, 0)},
                {"price": 110.0, "timestamp": datetime(2026, 1, 6, 15, 0, 0)},
            ],
            2: [
                {"price": 50.0, "timestamp": datetime(2026, 1, 5, 15, 0, 0)},
                {"price": 45.0, "timestamp": datetime(2026, 1, 6, 15, 0, 0)},
            ],
        }
        db = self._make_db(day_quotes=day_quotes)

        # 'gg': both tickers get buy signal
        with patch("pytink.experiments.prepare_data", return_value=_make_prepared()), \
             patch("pytink.experiments.train_and_evaluate", return_value=(0.5, 0.8)), \
             patch("pytink.experiments.predict_next_tokens", return_value=["gg"]), \
             patch("pytink.experiments.StockTransformerModel"):
            result = run_buy_hold_experiment(
                db=db, tickers=self.TICKERS, end_date=self.END_DATE,
                buy_target_dollars=5_000.0, device="cpu",
            )

        # A: 50 shares @ $100→$110, gain=$500
        # B: 100 shares @ $50→$45, gain=-$500
        assert result["buy_decisions"] == ["A", "B"]
        assert len(result["purchases"]) == 2
        assert result["total_gain"] == pytest.approx(0.0, abs=1e-6)

    # -- skip-stock paths ------------------------------------------------------

    def test_missing_buy_day_quote_skips_stock(self):
        """A stock is skipped when its buy-day quote is unavailable."""
        # Only a sell-day quote — no quote on end_date
        day_quotes = {
            1: [{"price": 105.0, "timestamp": datetime(2026, 1, 6, 15, 0, 0)}]
        }
        db = self._make_db(day_quotes=day_quotes)

        with patch("pytink.experiments.prepare_data", return_value=_make_prepared()), \
             patch("pytink.experiments.train_and_evaluate", return_value=(0.5, 0.8)), \
             patch("pytink.experiments.predict_next_tokens", return_value=["gd"]), \
             patch("pytink.experiments.StockTransformerModel"):
            result = run_buy_hold_experiment(
                db=db, tickers=self.TICKERS, end_date=self.END_DATE, device="cpu"
            )

        assert result["buy_decisions"] == ["A"]
        assert result["purchases"] == []
        assert result["total_gain"] == pytest.approx(0.0)

    def test_missing_sell_day_quote_skips_stock(self):
        """A stock is skipped when its sell-day quote is unavailable."""
        # Only a buy-day quote — no quote on sell_date
        day_quotes = {
            1: [{"price": 100.0, "timestamp": datetime(2026, 1, 5, 15, 0, 0)}]
        }
        db = self._make_db(day_quotes=day_quotes)

        with patch("pytink.experiments.prepare_data", return_value=_make_prepared()), \
             patch("pytink.experiments.train_and_evaluate", return_value=(0.5, 0.8)), \
             patch("pytink.experiments.predict_next_tokens", return_value=["gd"]), \
             patch("pytink.experiments.StockTransformerModel"):
            result = run_buy_hold_experiment(
                db=db, tickers=self.TICKERS, end_date=self.END_DATE, device="cpu"
            )

        assert result["buy_decisions"] == ["A"]
        assert result["purchases"] == []
        assert result["total_gain"] == pytest.approx(0.0)

    def test_no_day_quotes_at_all_skips_stock(self):
        """A stock is skipped when the day_quotes dict contains no entry for it."""
        day_quotes = {}  # no entries at all
        db = self._make_db(day_quotes=day_quotes)

        with patch("pytink.experiments.prepare_data", return_value=_make_prepared()), \
             patch("pytink.experiments.train_and_evaluate", return_value=(0.5, 0.8)), \
             patch("pytink.experiments.predict_next_tokens", return_value=["gd"]), \
             patch("pytink.experiments.StockTransformerModel"):
            result = run_buy_hold_experiment(
                db=db, tickers=self.TICKERS, end_date=self.END_DATE, device="cpu"
            )

        assert result["purchases"] == []

    # -- error paths -----------------------------------------------------------

    def test_raises_value_error_for_unknown_ticker(self):
        """ValueError is raised when a ticker is not found in the database."""
        db = MagicMock()
        # Only 'A' is returned; 'B' is missing
        db.get_stocks_by_tickers.return_value = [{"ticker": "A", "id": 1}]

        with pytest.raises(ValueError, match="B"):
            run_buy_hold_experiment(
                db=db, tickers=["A", "B"], end_date=self.END_DATE, device="cpu"
            )

    def test_raises_runtime_error_when_prepare_data_returns_none(self):
        """RuntimeError is raised when data is insufficient for model training."""
        db = MagicMock()
        db.get_stocks_by_tickers.return_value = self.STOCK_DICTS
        db.get_quotes_for_stocks.return_value = _make_training_quotes()

        with patch("pytink.experiments.prepare_data", return_value=None), \
             patch("pytink.experiments.StockTransformerModel"):
            with pytest.raises(RuntimeError):
                run_buy_hold_experiment(
                    db=db, tickers=self.TICKERS, end_date=self.END_DATE, device="cpu"
                )

    # -- result structure ------------------------------------------------------

    def test_result_dict_has_all_required_keys(self):
        """Result dict always contains every expected key."""
        db = self._make_db()

        with patch("pytink.experiments.prepare_data", return_value=_make_prepared()), \
             patch("pytink.experiments.train_and_evaluate", return_value=(0.5, 0.8)), \
             patch("pytink.experiments.predict_next_tokens", return_value=["dd"]), \
             patch("pytink.experiments.StockTransformerModel"):
            result = run_buy_hold_experiment(
                db=db, tickers=self.TICKERS, end_date=self.END_DATE, device="cpu"
            )

        required_keys = {
            "end_date", "tickers", "train_loss", "train_accuracy",
            "predicted_words", "buy_decisions", "purchases",
            "total_invested", "total_gain", "return_pct",
        }
        assert required_keys.issubset(result.keys())

    def test_predicted_words_forwarded_in_result(self):
        """The predicted_words returned by predict_next_tokens appear in the result."""
        db = self._make_db()
        expected = ["dd", "gg", "dd"]

        with patch("pytink.experiments.prepare_data", return_value=_make_prepared()), \
             patch("pytink.experiments.train_and_evaluate", return_value=(0.5, 0.8)), \
             patch("pytink.experiments.predict_next_tokens", return_value=expected), \
             patch("pytink.experiments.StockTransformerModel"):
            result = run_buy_hold_experiment(
                db=db, tickers=self.TICKERS, end_date=self.END_DATE, device="cpu"
            )

        assert result["predicted_words"] == expected
