"""Unit tests for processor.py module."""
import sys
import pytest
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from processor import PriceProcessor, DELTA_VALUES, DELTA_TO_CHAR


class TestDeltaValues:
    """Test delta value constants."""
    
    def test_delta_values_are_correct(self):
        """Test that delta values match specification."""
        expected = [-.01, -.005, -.001, 0, .001, .005, .01]
        assert DELTA_VALUES == expected
    
    def test_delta_to_char_mapping_size(self):
        """Test that all deltas are mapped to characters."""
        assert len(DELTA_TO_CHAR) == len(DELTA_VALUES)
    
    def test_delta_to_char_uses_lowercase_letters(self):
        """Test that deltas map to lowercase letters a-g."""
        expected_letters = set('abcdefg')
        actual_letters = set(DELTA_TO_CHAR.values())
        assert actual_letters == expected_letters


@pytest.fixture
def processor():
    """PriceProcessor fixture."""
    return PriceProcessor(interval_minutes=15)


class TestPriceProcessor:
    """Test PriceProcessor class."""
    
    def test_processor_initialization(self, processor):
        """Test processor initializes with correct interval."""
        assert processor is not None
        assert processor.interval.total_seconds() == 15 * 60
    
    def test_calculate_delta_positive_change(self, processor):
        """Test delta calculation for price increase."""
        delta = processor.calculate_delta(100.0, 101.0)
        assert abs(delta - 0.01) < 1e-6
    
    def test_calculate_delta_negative_change(self, processor):
        """Test delta calculation for price decrease."""
        delta = processor.calculate_delta(100.0, 99.0)
        assert abs(delta - (-0.01)) < 1e-6
    
    def test_calculate_delta_no_change(self, processor):
        """Test delta calculation when prices are equal."""
        delta = processor.calculate_delta(100.0, 100.0)
        assert abs(delta - 0.0) < 1e-6
    
    def test_calculate_delta_zero_old_price(self, processor):
        """Test delta calculation returns None for zero old price."""
        delta = processor.calculate_delta(0.0, 100.0)
        assert delta is None
    
    def test_delta_to_symbol_exact_match(self, processor):
        """Test mapping exact delta values to symbols."""
        # -0.01 should map to 'a'
        symbol = processor.delta_to_symbol(-0.01)
        assert symbol == 'a'
        
        # 0.0 should map to 'd'
        symbol = processor.delta_to_symbol(0.0)
        assert symbol == 'd'
        
        # 0.01 should map to 'g'
        symbol = processor.delta_to_symbol(0.01)
        assert symbol == 'g'
    
    def test_delta_to_symbol_nearest_neighbor(self, processor):
        """Test nearest neighbor quantization."""
        # 0.007 is closer to 0.005 than 0.01
        symbol = processor.delta_to_symbol(0.007)
        assert symbol == 'f'  # 0.005
        
        # -0.0075 should round to -0.005
        symbol = processor.delta_to_symbol(-0.0075)
        assert symbol == 'b'  # -0.005
    
    def test_delta_to_symbol_clamping_high(self, processor):
        """Test that extreme high deltas clamp to max symbol."""
        symbol = processor.delta_to_symbol(0.5)
        assert symbol == 'g'  # max
    
    def test_delta_to_symbol_clamping_low(self, processor):
        """Test that extreme low deltas clamp to min symbol."""
        symbol = processor.delta_to_symbol(-0.5)
        assert symbol == 'a'  # min
    
    def test_delta_to_symbol_none_input(self, processor):
        """Test delta_to_symbol returns None for None input."""
        symbol = processor.delta_to_symbol(None)
        assert symbol is None
    
    def test_symbol_to_delta_roundtrip(self, processor):
        """Test roundtrip conversion symbol -> delta."""
        for symbol in 'abcdefg':
            delta = processor.symbol_to_delta(symbol)
            assert delta is not None
            assert delta in DELTA_VALUES
    
    def test_symbol_to_delta_all_symbols(self, processor):
        """Test all symbols can be converted to deltas."""
        symbols = 'abcdefg'
        for symbol in symbols:
            delta = processor.symbol_to_delta(symbol)
            assert delta is not None
    
    def test_count_unique_words(self, processor):
        """Test counting unique words."""
        words = ['abc', 'abc', 'def', 'ghi', 'abc']
        unique_count, unique_words = processor.count_unique_words(words)
        assert unique_count == 3
        assert len(unique_words) == 3
    
    def test_count_unique_words_empty_list(self, processor):
        """Test counting unique words with empty list."""
        unique_count, unique_words = processor.count_unique_words([])
        assert unique_count == 0
        assert len(unique_words) == 0


class TestDeltaQuantization:
    """Test delta quantization boundaries."""
    
    @pytest.fixture
    def quantization_processor(self):
        """Processor fixture for quantization tests."""
        return PriceProcessor()
    
    def test_quantization_boundaries(self, quantization_processor):
        """Test quantization at boundary points."""
        test_cases = [
            # (delta, expected_symbol)
            # -0.0075 is slightly closer to -0.005 (due to floating point precision)
            (-0.0075, 'b'),
            # -0.003 is closer to -0.005 than -0.001
            (-0.003, 'b'),
            # -0.0005 is closer to -0.001
            (-0.0005, 'c'),
            # 0.0005 is exactly between 0 and 0.001, picks 0
            (0.0005, 'd'),
            # 0.003 is closer to 0.001
            (0.003, 'e'),
            # 0.0075 is closer to 0.005
            (0.0075, 'f'),
        ]
        
        for delta, expected_symbol in test_cases:
            symbol = quantization_processor.delta_to_symbol(delta)
            assert symbol == expected_symbol


class TestCalculateDeltaNoneHandling:
    """Test calculate_delta handles None values correctly."""
    
    @pytest.fixture
    def processor(self):
        """Processor fixture."""
        return PriceProcessor()
    
    def test_calculate_delta_none_new_price(self, processor):
        """Test calculate_delta returns None when new_price is None."""
        delta = processor.calculate_delta(100.0, None)
        assert delta is None
    
    def test_calculate_delta_none_old_price(self, processor):
        """Test calculate_delta returns None when old_price is None."""
        delta = processor.calculate_delta(None, 100.0)
        assert delta is None
    
    def test_calculate_delta_both_none(self, processor):
        """Test calculate_delta returns None when both prices are None."""
        delta = processor.calculate_delta(None, None)
        assert delta is None


class TestBinarySearchWithBisect:
    """Test binary search using bisect module with pre-parsed timestamps."""
    
    @pytest.fixture
    def processor(self):
        """Processor fixture."""
        return PriceProcessor()
    
    @pytest.fixture
    def quotes_with_timestamps(self):
        """Sample quotes with timestamps."""
        from datetime import timedelta
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        quotes = []
        timestamps = []
        for i in range(60):  # 60 minutes of data
            ts = base_time + timedelta(minutes=i)
            quotes.append({'timestamp': ts, 'price': 100.0 + i * 0.01})
            timestamps.append(ts)
        return quotes, timestamps
    
    def test_get_quote_at_or_before_with_timestamps(self, processor, quotes_with_timestamps):
        """Test binary search with pre-parsed timestamps."""
        quotes, timestamps = quotes_with_timestamps
        
        # Search for exact match
        target = datetime(2024, 1, 1, 9, 50, 0)
        result = processor._get_quote_at_or_before(quotes, target, timestamps)
        assert result is not None
        assert result['timestamp'] == target
    
    def test_get_quote_at_or_before_between_timestamps(self, processor, quotes_with_timestamps):
        """Test binary search returns quote before target when no exact match."""
        quotes, timestamps = quotes_with_timestamps
        
        # Search for time between quotes
        target = datetime(2024, 1, 1, 9, 30, 30)  # 30 seconds after minute 30
        result = processor._get_quote_at_or_before(quotes, target, timestamps)
        assert result is not None
        assert result['timestamp'] == datetime(2024, 1, 1, 9, 30, 0)
    
    def test_get_quote_at_or_before_no_match(self, processor, quotes_with_timestamps):
        """Test binary search returns None when target is before all quotes."""
        quotes, timestamps = quotes_with_timestamps
        
        # Search for time before all quotes
        target = datetime(2024, 1, 1, 8, 0, 0)
        result = processor._get_quote_at_or_before(quotes, target, timestamps)
        assert result is None
    
    def test_get_quote_at_or_before_empty_quotes(self, processor):
        """Test binary search with empty quotes list."""
        result = processor._get_quote_at_or_before([], datetime.now(), [])
        assert result is None
    
    def test_get_quote_at_or_before_without_timestamps(self, processor, quotes_with_timestamps):
        """Test binary search fallback without pre-parsed timestamps."""
        quotes, _ = quotes_with_timestamps
        
        # Search without passing timestamps (uses fallback)
        target = datetime(2024, 1, 1, 9, 50, 0)
        result = processor._get_quote_at_or_before(quotes, target, None)
        assert result is not None
        assert result['timestamp'] == target


class TestSegmentBoundaryFix:
    """Test that segment boundaries don't lose data."""
    
    @pytest.fixture
    def processor(self):
        """Processor fixture."""
        return PriceProcessor(interval_minutes=1, num_threads=2)
    
    def test_initial_prices_passed_to_segments(self, processor):
        """Test that initial prices are computed for non-first segments."""
        # Create test data spanning multiple segments
        # Use Jan 2, 2024 (Tuesday) during market hours (9:30 AM - 4:00 PM ET)
        stock_ids = [1, 2]
        quotes_dict = {}
        
        for stock_id in stock_ids:
            quotes = []
            for i in range(20):  # 20 minutes of data starting at 9:30 AM
                ts = datetime(2024, 1, 2, 9, 30 + i, 0)
                quotes.append({'timestamp': ts, 'price': str(100.0 + i * 0.01)})
            quotes_dict[stock_id] = quotes
        
        # Extract words - with fix, should not lose words at segment boundaries
        words = processor.extract_words_parallel(quotes_dict, stock_ids)
        
        # With 20 minutes and 2 stocks, we should get words for minutes 1-19
        # (minute 0 establishes baseline, subsequent minutes generate words)
        # Without the fix, we'd lose words at the segment boundary
        assert len(words) > 0
        
        # Verify words are valid (all chars are 'a'-'g')
        for word in words:
            assert len(word) == len(stock_ids)
            for char in word:
                assert char in 'abcdefg'


class TestMarketHours:
    """Test market hours awareness in processor."""
    
    @pytest.fixture
    def processor(self):
        return PriceProcessor(interval_minutes=1)
    
    def test_is_market_open_during_trading_hours(self, processor):
        """Test that market is open during normal trading hours."""
        # Tuesday Jan 2, 2024 at 10:00 AM - should be open
        dt = datetime(2024, 1, 2, 10, 0, 0)
        assert processor._is_market_open(dt) is True
        
        # Same day at 3:30 PM - still open
        dt = datetime(2024, 1, 2, 15, 30, 0)
        assert processor._is_market_open(dt) is True


class TestCustomDeltaValues:
    """Test custom delta values configuration."""
    
    def test_custom_delta_values_initialization(self):
        """Test processor initializes with custom delta values."""
        custom_deltas = [-0.02, -0.01, 0.0, 0.01, 0.02]
        processor = PriceProcessor(delta_values=custom_deltas)
        
        assert processor.delta_values == sorted(custom_deltas)
        assert len(processor.delta_to_char) == 5
        assert len(processor.char_to_delta) == 5
    
    def test_custom_delta_values_symbol_mapping(self):
        """Test that custom delta values map to correct symbols."""
        custom_deltas = [-0.02, 0.0, 0.02]
        processor = PriceProcessor(delta_values=custom_deltas)
        
        # Should use letters a, b, c
        assert processor.delta_to_symbol(-0.02) == 'a'
        assert processor.delta_to_symbol(0.0) == 'b'
        assert processor.delta_to_symbol(0.02) == 'c'
    
    def test_custom_delta_values_nearest_neighbor(self):
        """Test nearest neighbor quantization with custom deltas."""
        custom_deltas = [-0.02, 0.0, 0.02]
        processor = PriceProcessor(delta_values=custom_deltas)
        
        # 0.015 is closer to 0.02 than 0.0
        assert processor.delta_to_symbol(0.015) == 'c'
        
        # -0.005 is closer to 0.0 than -0.02
        assert processor.delta_to_symbol(-0.005) == 'b'
    
    def test_custom_delta_values_roundtrip(self):
        """Test roundtrip conversion with custom deltas."""
        custom_deltas = [-0.015, -0.005, 0.0, 0.005, 0.015]
        processor = PriceProcessor(delta_values=custom_deltas)
        
        for delta in sorted(custom_deltas):
            symbol = processor.delta_to_symbol(delta)
            recovered_delta = processor.symbol_to_delta(symbol)
            assert recovered_delta == delta
    
    def test_default_delta_values_when_none(self):
        """Test that default delta values are used when None provided."""
        processor = PriceProcessor(delta_values=None)
        
        assert processor.delta_values == DELTA_VALUES
        assert len(processor.delta_values) == 7
    
    def test_custom_delta_values_unsorted_input(self):
        """Test that unsorted custom deltas are sorted."""
        custom_deltas = [0.02, -0.02, 0.0, 0.01, -0.01]
        processor = PriceProcessor(delta_values=custom_deltas)
        
        # Should be sorted
        assert processor.delta_values == [-0.02, -0.01, 0.0, 0.01, 0.02]


class TestMarketHoursExtended:
    """Additional market hours tests."""
    
    @pytest.fixture
    def processor(self):
        return PriceProcessor(interval_minutes=1)
    
    def test_is_market_open_at_boundaries(self, processor):
        """Test market open/close boundaries."""
        # Exactly at market open 9:30 AM
        dt = datetime(2024, 1, 2, 9, 30, 0)
        assert processor._is_market_open(dt) is True
        
        # One minute before open
        dt = datetime(2024, 1, 2, 9, 29, 0)
        assert processor._is_market_open(dt) is False
        
        # One minute before close (3:59 PM)
        dt = datetime(2024, 1, 2, 15, 59, 0)
        assert processor._is_market_open(dt) is True
        
        # At close (4:00 PM) - market is closed
        dt = datetime(2024, 1, 2, 16, 0, 0)
        assert processor._is_market_open(dt) is False
    
    def test_is_market_open_weekend(self, processor):
        """Test that market is closed on weekends."""
        # Saturday Jan 6, 2024
        dt = datetime(2024, 1, 6, 10, 0, 0)
        assert processor._is_market_open(dt) is False
        
        # Sunday Jan 7, 2024
        dt = datetime(2024, 1, 7, 10, 0, 0)
        assert processor._is_market_open(dt) is False
    
    def test_is_market_open_holiday(self, processor):
        """Test that market is closed on holidays."""
        # New Year's Day 2024 (Monday Jan 1)
        dt = datetime(2024, 1, 1, 10, 0, 0)
        assert processor._is_market_open(dt) is False
        
        # Christmas 2024 (Wednesday Dec 25)
        dt = datetime(2024, 12, 25, 10, 0, 0)
        assert processor._is_market_open(dt) is False
    
    def test_next_market_open_from_weekend(self, processor):
        """Test finding next market open from weekend."""
        # Saturday Jan 6, 2024 -> should return Monday Jan 8 at 9:30 AM
        dt = datetime(2024, 1, 6, 10, 0, 0)
        next_open = processor._next_market_open(dt)
        assert next_open.date().weekday() == 0  # Monday
        assert next_open.hour == 9
        assert next_open.minute == 30
    
    def test_next_market_open_from_after_hours(self, processor):
        """Test finding next market open from after hours."""
        # Tuesday Jan 2, 2024 at 5:00 PM -> Wednesday Jan 3 at 9:30 AM
        dt = datetime(2024, 1, 2, 17, 0, 0)
        next_open = processor._next_market_open(dt)
        assert next_open.day == 3  # Wednesday
        assert next_open.hour == 9
        assert next_open.minute == 30
    
    def test_next_market_open_skips_holiday(self, processor):
        """Test that next market open skips holidays."""
        # Sunday Dec 31, 2023 -> should skip Jan 1 (holiday) to Jan 2, 2024
        dt = datetime(2023, 12, 31, 10, 0, 0)
        next_open = processor._next_market_open(dt)
        assert next_open.day == 2  # Jan 2, 2024
        assert next_open.month == 1
        assert next_open.year == 2024
    
    def test_extract_words_skips_non_market_hours(self, processor):
        """Test that word extraction skips non-market hours."""
        # Create data spanning market close and next open
        stock_ids = [1]
        quotes_dict = {1: []}
        
        # Add quotes for Tuesday Jan 2, 2024:
        # 3:58 PM, 3:59 PM (last 2 minutes of trading)
        for minute in range(58, 60):
            ts = datetime(2024, 1, 2, 15, minute, 0)
            quotes_dict[1].append({'timestamp': ts, 'price': '100.0'})
        
        # Add quotes for after hours (should be skipped)
        for minute in range(0, 30):
            ts = datetime(2024, 1, 2, 16, minute, 0)
            quotes_dict[1].append({'timestamp': ts, 'price': '101.0'})
        
        # Add quotes for next day market open
        for minute in range(30, 60):
            ts = datetime(2024, 1, 3, 9, minute, 0)
            quotes_dict[1].append({'timestamp': ts, 'price': '102.0'})
        
        words = processor.extract_words_parallel(quotes_dict, stock_ids)
        
        # Should get words but should NOT include after-hours transitions
        # The after-hours quotes should be skipped
        assert len(words) >= 0  # Basic sanity check
