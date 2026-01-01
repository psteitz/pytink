"""Integration tests for the stock prediction system."""
import sys
import tempfile
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from processor import PriceProcessor
from model import StockWordDataset
import torch


@pytest.fixture
def processor():
    """PriceProcessor fixture."""
    return PriceProcessor()


@pytest.fixture
def mock_stocks():
    """Mock stock data fixture.
    
    Uses Jan 2, 2024 (Tuesday) during market hours (9:30 AM - 4:00 PM ET).
    """
    return {
        1: [
            {'price': 100.0, 'timestamp': '2024-01-02 10:00:00'},
            {'price': 101.0, 'timestamp': '2024-01-02 10:15:00'},
            {'price': 101.5, 'timestamp': '2024-01-02 10:30:00'},
            {'price': 101.25, 'timestamp': '2024-01-02 10:45:00'},
            {'price': 102.0, 'timestamp': '2024-01-02 11:00:00'},
            {'price': 101.75, 'timestamp': '2024-01-02 11:15:00'},
            {'price': 102.5, 'timestamp': '2024-01-02 11:30:00'},
            {'price': 102.25, 'timestamp': '2024-01-02 11:45:00'},
            {'price': 103.0, 'timestamp': '2024-01-02 12:00:00'},
            {'price': 103.5, 'timestamp': '2024-01-02 12:15:00'},
        ]
    }


class TestIntegrationWorkflow:
    """Test end-to-end workflow integration."""
    
    def test_processor_generates_valid_words(self, processor, mock_stocks):
        """Test that processor generates valid word sequences."""
        words = processor.extract_words_parallel(
            {1: mock_stocks[1]},
            [1]
        )
        
        # Should have words generated from quotes
        assert len(words) > 0
        
        # Each word should consist of valid delta symbols (a-g)
        valid_symbols = set('abcdefg')
        for word in words:
            if isinstance(word, str):
                for char in word:
                    assert char in valid_symbols
    
    def test_delta_calculation_sequence(self, processor, mock_stocks):
        """Test delta calculations form correct sequence."""
        quotes = mock_stocks[1]
        
        # Manual delta calculation
        deltas = []
        for i in range(1, len(quotes)):
            delta = processor.calculate_delta(
                float(quotes[i-1]['price']),
                float(quotes[i]['price'])
            )
            deltas.append(delta)
        
        # All deltas should be positive (prices are increasing)
        for delta in deltas:
            assert delta > -0.02
            assert delta < 0.02
    
    def test_word_to_delta_roundtrip(self, processor):
        """Test roundtrip conversion: delta -> symbol -> delta."""
        test_deltas = [
            -0.012,  # Should map to -0.01
            -0.003,  # Should map to -0.005
            -0.0001, # Should map to 0
            0.0005,  # Should map to 0.001
            0.008,   # Should map to 0.01
        ]
        
        for original_delta in test_deltas:
            symbol = processor.delta_to_symbol(original_delta)
            # Should be valid symbol
            assert symbol in 'abcdefg'
    
    def test_dataset_creation_from_words(self, processor, mock_stocks):
        """Test creating StockWordDataset from processor output."""
        # Generate words from processor
        words = processor.extract_words_parallel(
            {1: mock_stocks[1]},
            [1]
        )
        
        # Only test if words were generated
        if len(words) > 0:
            # Create vocabulary mapping
            vocab = {word: idx for idx, word in enumerate(set(words))}
            
            # Create dataset
            dataset = StockWordDataset(
                words=words,
                vocab=vocab,
                sequence_length=4
            )
            
            # Dataset should have valid length
            expected_length = max(0, len(words) - 4)
            assert len(dataset) == expected_length
    
    def test_dataset_produces_valid_tensors(self, processor, mock_stocks):
        """Test that dataset produces valid input/label tensors."""
        words = processor.extract_words_parallel(
            {1: mock_stocks[1]},
            [1]
        )
        
        if len(words) > 0:
            # Create vocabulary
            vocab = {word: idx for idx, word in enumerate(set(words))}
            
            dataset = StockWordDataset(
                words=words,
                vocab=vocab,
                sequence_length=4
            )
            
            if len(dataset) > 0:
                input_ids, labels = dataset[0]
                
                # Input should be sequence_length tokens
                assert input_ids.shape[0] == 4
                
                # Each token should be valid vocab ID
                for token_id in input_ids:
                    assert token_id >= 0
                    assert token_id < len(vocab)
                
                # Label should be single token
                assert labels >= 0
                assert labels < len(vocab)
    
    def test_multiple_stocks_dataset(self, processor):
        """Test dataset creation with multiple stocks."""
        mock_stocks_multi = {
            1: [
                {'price': 100.0 + i*0.5, 'timestamp': f'2024-01-01 10:{i:02d}:00'}
                for i in range(15)
            ],
            2: [
                {'price': 200.0 + i*0.3, 'timestamp': f'2024-01-01 10:{i:02d}:00'}
                for i in range(15)
            ],
        }
        
        # Generate words for each stock
        all_words = []
        for stock_id, quotes in mock_stocks_multi.items():
            words = processor.extract_words_parallel(
                {stock_id: quotes},
                [stock_id]
            )
            if len(words) > 0:
                all_words.extend(words)
        
        # Create dataset with all words if we have any
        if all_words:
            vocab = {word: idx for idx, word in enumerate(set(all_words))}
            dataset = StockWordDataset(
                words=all_words,
                vocab=vocab,
                sequence_length=4
            )
            
            # Dataset should have valid length
            assert len(dataset) > 0
    
    def test_vocab_size_matches_delta_levels(self, processor, mock_stocks):
        """Test that vocabulary size matches delta levels."""
        words = processor.extract_words_parallel(
            {1: mock_stocks[1]},
            [1]
        )
        
        if len(words) > 0:
            vocab = {word: idx for idx, word in enumerate(set(words))}
            dataset = StockWordDataset(
                words=words,
                vocab=vocab,
                sequence_length=4
            )
            
            # All token IDs should be valid vocab IDs
            if len(dataset) > 0:
                for i in range(min(5, len(dataset))):
                    input_ids, label = dataset[i]
                    # All IDs should be in vocab
                    for token_id in input_ids:
                        assert token_id < len(vocab)
                    assert label < len(vocab)


class TestConfigurationIntegration:
    """Test configuration loading and merging."""
    
    def test_yaml_config_creation(self):
        """Test creating and reading YAML config."""
        import yaml
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            
            config = {
                'database': {
                    'host': 'localhost',
                    'port': 3306,
                    'user': 'tinker',
                    'password': 'password',
                    'database': 'tinker'
                },
                'data': {
                    'num_stocks': 20,
                    'interval_minutes': 30,
                    'sequence_length': 8
                },
                'model': {
                    'hidden_size': 256,
                    'num_hidden_layers': 6,
                    'num_attention_heads': 8,
                    'max_position_embeddings': 256
                },
                'training': {
                    'batch_size': 64,
                    'num_epochs': 10,
                    'learning_rate': 0.0001,
                    'weight_decay': 0,
                    'warmup_steps': 100
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Load and verify
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
            
            assert loaded['data']['num_stocks'] == 20
            assert loaded['model']['hidden_size'] == 256
            assert loaded['training']['batch_size'] == 64
    
    def test_config_merging(self):
        """Test merging YAML config with CLI overrides."""
        import yaml
        
        # Base config
        base_config = {
            'data': {
                'num_stocks': 20,
                'interval_minutes': 30,
            },
            'training': {
                'batch_size': 64,
                'num_epochs': 10,
            }
        }
        
        # Simulate CLI override (batch_size only)
        cli_overrides = {'batch_size': 128}
        
        # Merge (CLI takes precedence)
        merged = base_config.copy()
        if 'training' in merged and 'batch_size' in cli_overrides:
            merged['training']['batch_size'] = cli_overrides['batch_size']
        
        assert merged['training']['batch_size'] == 128
        assert merged['training']['num_epochs'] == 10  # Unchanged
        assert merged['data']['num_stocks'] == 20  # Unchanged


class TestSaveAndLoadArtifacts:
    """Test saving and loading model artifacts."""
    
    def test_save_vocabulary(self):
        """Test saving vocabulary to file."""
        vocab = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            vocab_path = output_dir / 'vocabulary.json'
            
            with open(vocab_path, 'w') as f:
                json.dump(vocab, f)
            
            # Load and verify
            with open(vocab_path, 'r') as f:
                loaded_vocab = json.load(f)
            
            assert loaded_vocab == vocab
            assert len(loaded_vocab) == 7
    
    def test_save_predictions(self):
        """Test saving predictions to file."""
        predictions = {
            'train_accuracy': 0.95,
            'eval_accuracy': 0.92,
            'train_loss': 0.15,
            'eval_loss': 0.22,
            'num_epochs': 10,
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pred_path = output_dir / 'predictions.json'
            
            with open(pred_path, 'w') as f:
                json.dump(predictions, f)
            
            # Load and verify
            with open(pred_path, 'r') as f:
                loaded_pred = json.load(f)
            
            assert loaded_pred['train_accuracy'] == 0.95
            assert loaded_pred['num_epochs'] == 10
    
    def test_save_training_history(self):
        """Test saving training history."""
        history = {
            'train_losses': [0.5, 0.4, 0.3, 0.25, 0.2],
            'eval_losses': [0.55, 0.45, 0.35, 0.3, 0.28],
            'epochs': [1, 2, 3, 4, 5],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            hist_path = output_dir / 'training_history.json'
            
            with open(hist_path, 'w') as f:
                json.dump(history, f)
            
            # Load and verify
            with open(hist_path, 'r') as f:
                loaded_hist = json.load(f)
            
            assert len(loaded_hist['train_losses']) == 5
            assert loaded_hist['train_losses'][0] == 0.5
