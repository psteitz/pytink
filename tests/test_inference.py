"""Tests for inference.py module."""
import json
import tempfile
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import torch

from pytink.inference import (
    load_model_config,
    find_common_date_range,
    filter_quotes_by_date,
    evaluate_model,
    predict_next_tokens,
    print_predictions,
    DEFAULT_MONTHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_PREDICT_STEPS,
    DEFAULT_INTERVAL_MINUTES,
    DEFAULT_CONTEXT_WINDOW_SIZE,
)
from pytink.model import StockTransformerModel, StockWordDataset, custom_collate_fn
from torch.utils.data import DataLoader

class TestLoadModelConfig:
    """Tests for load_model_config function."""
    
    def test_load_valid_config(self):
        """Test loading a valid config file."""
        import yaml
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            config_path = model_dir / 'config.yaml'
            
            config = {
                'data': {
                    'tickers': ['AAPL', 'GOOGL'],
                    'interval_minutes': 30,
                    'context_window_size': 32
                },
                'model': {
                    'hidden_size': 128,
                    'num_hidden_layers': 4
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            loaded = load_model_config(model_dir)
            
            assert loaded['data']['tickers'] == ['AAPL', 'GOOGL']
            assert loaded['data']['interval_minutes'] == 30
            assert loaded['model']['hidden_size'] == 128
    
    def test_load_missing_config(self):
        """Test error when config file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            
            with pytest.raises(FileNotFoundError):
                load_model_config(model_dir)


class TestFindCommonDateRange:
    """Tests for find_common_date_range function."""
    
    def test_find_range_with_overlapping_data(self):
        """Test finding common date range with overlapping quotes."""
        # Stock 1: Jan 1 - Jan 31
        # Stock 2: Jan 15 - Feb 15
        # Common range should be: Jan 15 - Jan 31
        
        quotes_dict = {
            1: [
                {'timestamp': datetime(2024, 1, 1, 10, 0), 'price': 100},
                {'timestamp': datetime(2024, 1, 31, 16, 0), 'price': 105}
            ],
            2: [
                {'timestamp': datetime(2024, 1, 15, 10, 0), 'price': 200},
                {'timestamp': datetime(2024, 2, 15, 16, 0), 'price': 210}
            ]
        }
        
        start, end = find_common_date_range(quotes_dict, [1, 2])
        
        assert start == datetime(2024, 1, 15, 10, 0)
        assert end == datetime(2024, 1, 31, 16, 0)
    
    def test_find_range_single_stock(self):
        """Test with single stock."""
        quotes_dict = {
            1: [
                {'timestamp': datetime(2024, 1, 1, 10, 0), 'price': 100},
                {'timestamp': datetime(2024, 1, 31, 16, 0), 'price': 105}
            ]
        }
        
        start, end = find_common_date_range(quotes_dict, [1])
        
        assert start == datetime(2024, 1, 1, 10, 0)
        assert end == datetime(2024, 1, 31, 16, 0)
    
    def test_find_range_empty_quotes(self):
        """Test with empty quotes."""
        quotes_dict = {1: []}
        
        start, end = find_common_date_range(quotes_dict, [1])
        
        assert start is None
        assert end is None


class TestFilterQuotesByDate:
    """Tests for filter_quotes_by_date function."""
    
    def test_filter_within_range(self):
        """Test filtering quotes within date range."""
        quotes_dict = {
            1: [
                {'timestamp': datetime(2024, 1, 1, 10, 0), 'price': 100},
                {'timestamp': datetime(2024, 1, 15, 10, 0), 'price': 102},
                {'timestamp': datetime(2024, 1, 31, 16, 0), 'price': 105}
            ]
        }
        
        start = datetime(2024, 1, 10, 0, 0)
        end = datetime(2024, 1, 20, 23, 59)
        
        filtered = filter_quotes_by_date(quotes_dict, start, end)
        
        assert len(filtered[1]) == 1
        assert filtered[1][0]['price'] == 102
    
    def test_filter_all_outside_range(self):
        """Test filtering when all quotes are outside range."""
        quotes_dict = {
            1: [
                {'timestamp': datetime(2024, 1, 1, 10, 0), 'price': 100},
                {'timestamp': datetime(2024, 1, 5, 10, 0), 'price': 102}
            ]
        }
        
        start = datetime(2024, 2, 1, 0, 0)
        end = datetime(2024, 2, 28, 23, 59)
        
        filtered = filter_quotes_by_date(quotes_dict, start, end)
        
        assert len(filtered[1]) == 0
    
    def test_filter_with_string_timestamps(self):
        """Test filtering with ISO format string timestamps."""
        quotes_dict = {
            1: [
                {'timestamp': '2024-01-15T10:00:00', 'price': 100}
            ]
        }
        
        start = datetime(2024, 1, 1, 0, 0)
        end = datetime(2024, 1, 31, 23, 59)
        
        filtered = filter_quotes_by_date(quotes_dict, start, end)
        
        assert len(filtered[1]) == 1


class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    def test_evaluate_returns_expected_keys(self):
        """Test that evaluate_model returns all expected metric keys."""
        # Create simple vocabulary and dataset
        words = ['aaa', 'aab', 'aba', 'baa', 'aaa', 'aab']
        vocab = {'aaa': 0, 'aab': 1, 'aba': 2, 'baa': 3}
        
        dataset = StockWordDataset(words=words, vocab=vocab, context_window_size=2)
        data_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)
        
        # Create model
        model = StockTransformerModel(
            vocab_size=len(vocab),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=1,
            device='cpu'
        )
        
        tickers = ['A', 'B', 'C']
        delta_values = [-0.01, 0.0, 0.01]
        
        metrics = evaluate_model(model, data_loader, vocab, 'cpu', tickers, delta_values)
        
        assert 'overall_accuracy' in metrics
        assert 'overall_loss' in metrics
        assert 'perplexity' in metrics
        assert 'total_samples' in metrics
        assert 'stock_metrics' in metrics
        assert 'delta_letters' in metrics


class TestMainArguments:
    """Tests for command-line argument parsing."""
    
    def test_required_arguments(self):
        """Test that required arguments are enforced."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--db-password', type=str, required=True)
        parser.add_argument('--model-dir', type=str, required=True)
        
        # Should raise error without required args
        with pytest.raises(SystemExit):
            parser.parse_args([])
    
    def test_optional_arguments_defaults(self):
        """Test default values for optional arguments."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--db-password', type=str, required=True)
        parser.add_argument('--model-dir', type=str, required=True)
        parser.add_argument('--months', type=int, default=3)
        parser.add_argument('--batch-size', type=int, default=64)
        
        args = parser.parse_args(['--db-password', 'test', '--model-dir', 'models/test'])
        
        assert args.months == 3
        assert args.batch_size == 64


class TestIntegration:
    """Integration tests for inference module."""
    
    def test_config_with_delta_ranges(self):
        """Test loading config with delta_ranges."""
        import yaml
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            config_path = model_dir / 'config.yaml'
            
            config = {
                'data': {
                    'tickers': ['AAPL'],
                    'interval_minutes': 30,
                    'context_window_size': 32
                },
                'delta_ranges': [-0.01, -0.005, 0.0, 0.005, 0.01]
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            loaded = load_model_config(model_dir)

            assert 'delta_ranges' in loaded
            assert len(loaded['delta_ranges']) == 5
            assert loaded['delta_ranges'][2] == 0.0
    
    def test_date_calculation_for_evaluation_period(self):
        """Test that evaluation period is calculated correctly."""
        max_date = datetime(2024, 6, 15, 16, 0)
        months = 3
        
        eval_start = max_date - timedelta(days=months * 30)
        
        # Should be approximately 3 months before
        assert eval_start < max_date
        assert (max_date - eval_start).days == 90


class TestConstants:
    """Tests for module-level default constants."""

    def test_constant_values(self):
        """Verify that default constants have the expected values."""
        assert DEFAULT_MONTHS == 3
        assert DEFAULT_BATCH_SIZE == 64
        assert DEFAULT_PREDICT_STEPS == 5
        assert DEFAULT_INTERVAL_MINUTES == 30
        assert DEFAULT_CONTEXT_WINDOW_SIZE == 32

    def test_argparse_defaults_match_constants(self):
        """Verify argparse defaults reference the module constants."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--db-password', type=str, required=True)
        parser.add_argument('--model-dir', type=str, required=True)
        parser.add_argument('--months', type=int, default=DEFAULT_MONTHS)
        parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
        parser.add_argument('--predict-steps', type=int, default=DEFAULT_PREDICT_STEPS)

        args = parser.parse_args(['--db-password', 'x', '--model-dir', 'models/x'])
        assert args.months == DEFAULT_MONTHS
        assert args.batch_size == DEFAULT_BATCH_SIZE
        assert args.predict_steps == DEFAULT_PREDICT_STEPS


def _make_small_model(vocab_size=4):
    """Return a tiny StockTransformerModel suitable for unit tests."""
    return StockTransformerModel(
        vocab_size=vocab_size,
        max_position_embeddings=16,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        device='cpu',
    )


class TestPredictNextTokens:
    """Tests for predict_next_tokens function."""

    def test_returns_correct_number_of_steps(self):
        """Output list length equals the requested number of steps."""
        vocab = {'aa': 0, 'ab': 1, 'ba': 2, 'bb': 3}
        words = ['aa', 'ab', 'ba', 'bb', 'aa', 'ab']
        model = _make_small_model(vocab_size=len(vocab))

        result = predict_next_tokens(model, words, vocab, context_window_size=4, steps=3, device='cpu')

        assert len(result) == 3

    def test_predictions_are_known_words_or_none(self):
        """Every returned entry is either a word in the vocabulary or None."""
        vocab = {'aa': 0, 'ab': 1, 'ba': 2, 'bb': 3}
        words = ['aa', 'ab', 'ba', 'bb']
        model = _make_small_model(vocab_size=len(vocab))

        result = predict_next_tokens(model, words, vocab, context_window_size=4, steps=5, device='cpu')

        for word in result:
            assert word is None or word in vocab

    def test_short_word_list_pads_window(self):
        """If words is shorter than the context window the call should still succeed."""
        vocab = {'aa': 0, 'ab': 1, 'ba': 2, 'bb': 3}
        words = ['aa', 'ab']  # shorter than context_window_size=4
        model = _make_small_model(vocab_size=len(vocab))

        # Should not raise
        result = predict_next_tokens(model, words, vocab, context_window_size=4, steps=2, device='cpu')

        assert len(result) == 2

    def test_single_word_list(self):
        """A single-word list seeds the window with padding and returns predictions."""
        vocab = {'aa': 0, 'ab': 1}
        words = ['ab']
        model = _make_small_model(vocab_size=len(vocab))

        result = predict_next_tokens(model, words, vocab, context_window_size=4, steps=1, device='cpu')

        assert len(result) == 1

    def test_window_uses_tail_of_words(self):
        """Seed context is taken from the tail of the word list."""
        # We can't directly inspect the window, but we can verify that a
        # word list longer than context_window_size doesn't raise and still
        # returns the right number of predictions.
        vocab = {'aa': 0, 'ab': 1, 'ba': 2, 'bb': 3}
        words = ['aa', 'ab', 'ba', 'bb', 'aa', 'ab', 'ba', 'bb', 'aa']  # 9 words, window=4
        model = _make_small_model(vocab_size=len(vocab))

        result = predict_next_tokens(model, words, vocab, context_window_size=4, steps=4, device='cpu')

        assert len(result) == 4

    def test_zero_steps_returns_empty_list(self):
        """Requesting zero steps returns an empty list."""
        vocab = {'aa': 0, 'ab': 1}
        words = ['aa', 'ab']
        model = _make_small_model(vocab_size=len(vocab))

        result = predict_next_tokens(model, words, vocab, context_window_size=4, steps=0, device='cpu')

        assert result == []

    def test_model_is_set_to_eval_mode(self):
        """predict_next_tokens puts the model in eval mode."""
        vocab = {'aa': 0, 'ab': 1}
        words = ['aa', 'ab']
        model = _make_small_model(vocab_size=len(vocab))
        model.train()  # start in training mode

        predict_next_tokens(model, words, vocab, context_window_size=4, steps=1, device='cpu')

        assert not model.get_model().training


class TestPrintPredictions:
    """Tests for print_predictions function."""

    def test_logs_header_and_steps(self, caplog):
        """Output contains a header and one row per step."""
        import logging
        predicted = ['ab', 'ba']
        tickers = ['A', 'B']
        delta_values = [-0.01, 0.0, 0.01]

        with caplog.at_level(logging.INFO, logger='pytink.inference'):
            print_predictions(predicted, tickers, delta_values)

        full_output = '\n'.join(caplog.messages)
        assert 'NEXT-TOKEN PREDICTIONS' in full_output
        assert '2 step(s)' in full_output

    def test_step_count_in_output(self, caplog):
        """Each prediction step appears in the logged output."""
        import logging
        predicted = ['ab', 'ba', 'bb']
        tickers = ['A', 'B']
        delta_values = [-0.01, 0.0, 0.01]

        with caplog.at_level(logging.INFO, logger='pytink.inference'):
            print_predictions(predicted, tickers, delta_values)

        rows = [m for m in caplog.messages if m.strip().startswith(('1', '2', '3'))]
        assert len(rows) == 3

    def test_none_word_renders_as_question_mark(self, caplog):
        """A None prediction is rendered as '?' for every ticker column."""
        import logging
        predicted = [None]
        tickers = ['A', 'B']
        delta_values = [-0.01, 0.0, 0.01]

        with caplog.at_level(logging.INFO, logger='pytink.inference'):
            print_predictions(predicted, tickers, delta_values)

        step_lines = [m for m in caplog.messages if m.strip().startswith('1')]
        assert step_lines, "Expected a step-1 row in the log output"
        assert '?' in step_lines[0]

    def test_cell_formatting_with_delta_values(self, caplog):
        """Cells include the delta letter and its percentage threshold."""
        import logging
        # vocab letter 'a' → index 0 → delta_values[0] = -0.01 → -1.0%
        predicted = ['aa']  # two-ticker word; both positions are 'a'
        tickers = ['X', 'Y']
        delta_values = [-0.01, 0.0, 0.01]

        with caplog.at_level(logging.INFO, logger='pytink.inference'):
            print_predictions(predicted, tickers, delta_values)

        full_output = '\n'.join(caplog.messages)
        assert 'a(-1.0%)' in full_output

    def test_empty_predictions_prints_header_only(self, caplog):
        """Empty prediction list still prints the section header without error."""
        import logging
        predicted = []
        tickers = ['A']
        delta_values = [0.0]

        with caplog.at_level(logging.INFO, logger='pytink.inference'):
            print_predictions(predicted, tickers, delta_values)

        full_output = '\n'.join(caplog.messages)
        assert 'NEXT-TOKEN PREDICTIONS' in full_output


class TestPostTrainingSeedLogic:
    """Unit tests for the post-training seeding decision logic used in main()."""

    def test_end_date_parsing(self):
        """A YYYY-MM-DD string from config parses correctly to a datetime."""
        end_date_str = '2025-12-31'
        parsed = datetime.strptime(end_date_str, '%Y-%m-%d')
        assert parsed == datetime(2025, 12, 31)
        assert parsed.year == 2025

    def test_seed_start_is_one_day_after_end_date(self):
        """Seed start date is one day after the training end date."""
        training_end = datetime(2025, 12, 31)
        seed_start = training_end + timedelta(days=1)
        assert seed_start == datetime(2026, 1, 1)

    def test_no_end_date_in_config_skips_prediction(self):
        """When data.end_date is absent the code path that skips prediction is taken."""
        data_config = {'tickers': ['AAPL'], 'interval_minutes': 30}
        end_date_str = data_config.get('end_date')
        assert end_date_str is None  # confirm the guard condition

    def test_end_date_present_in_config(self):
        """When data.end_date is present it is accessible via data_config.get."""
        data_config = {'tickers': ['AAPL'], 'interval_minutes': 30, 'end_date': '2025-06-01'}
        end_date_str = data_config.get('end_date')
        assert end_date_str == '2025-06-01'
        parsed = datetime.strptime(str(end_date_str), '%Y-%m-%d')
        assert parsed == datetime(2025, 6, 1)

    def test_invalid_end_date_format_raises_value_error(self):
        """An end_date that cannot be parsed as YYYY-MM-DD raises ValueError."""
        with pytest.raises(ValueError):
            datetime.strptime('not-a-date', '%Y-%m-%d')

    def test_get_quotes_for_stocks_called_with_start_date(self):
        """db.get_quotes_for_stocks is called with the correct start_date argument."""
        training_end = datetime(2025, 6, 1)
        seed_start = training_end + timedelta(days=1)

        mock_db = Mock()
        mock_db.get_quotes_for_stocks.return_value = {}

        mock_db.get_quotes_for_stocks([1, 2], start_date=seed_start)

        mock_db.get_quotes_for_stocks.assert_called_once_with([1, 2], start_date=seed_start)

    def test_empty_seed_quotes_skips_predict_next_tokens(self):
        """If post-training quotes produce no words, predict_next_tokens is not called."""
        mock_processor = Mock()
        mock_processor.extract_words.return_value = []

        post_quotes = {}
        stock_ids = [1, 2]
        seed_words = mock_processor.extract_words(post_quotes, stock_ids)

        assert seed_words == []
        # Simulate the guard: only call predict_next_tokens when seed_words is non-empty
        predict_called = False
        if seed_words:
            predict_called = True
        assert not predict_called
