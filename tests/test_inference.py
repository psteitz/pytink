"""Tests for inference.py module."""
import sys
import json
import tempfile
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch


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
            
            # Import after creating file
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from inference import load_model_config
            
            loaded = load_model_config(model_dir)
            
            assert loaded['data']['tickers'] == ['AAPL', 'GOOGL']
            assert loaded['data']['interval_minutes'] == 30
            assert loaded['model']['hidden_size'] == 128
    
    def test_load_missing_config(self):
        """Test error when config file is missing."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference import load_model_config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            
            with pytest.raises(FileNotFoundError):
                load_model_config(model_dir)


class TestFindCommonDateRange:
    """Tests for find_common_date_range function."""
    
    def test_find_range_with_overlapping_data(self):
        """Test finding common date range with overlapping quotes."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference import find_common_date_range
        
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
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference import find_common_date_range
        
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
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference import find_common_date_range
        
        quotes_dict = {1: []}
        
        start, end = find_common_date_range(quotes_dict, [1])
        
        assert start is None
        assert end is None


class TestFilterQuotesByDate:
    """Tests for filter_quotes_by_date function."""
    
    def test_filter_within_range(self):
        """Test filtering quotes within date range."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference import filter_quotes_by_date
        
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
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference import filter_quotes_by_date
        
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
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference import filter_quotes_by_date
        
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
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference import evaluate_model
        from model import StockTransformerModel, StockWordDataset, custom_collate_fn
        from torch.utils.data import DataLoader
        
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
            
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from inference import load_model_config
            
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
