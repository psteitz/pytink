"""Unit tests for model.py module."""
import sys
import pytest
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import StockWordDataset, StockTransformerModel, custom_collate_fn


@pytest.fixture
def words():
    """Sample words fixture."""
    return ['abc', 'def', 'ghi', 'jkl', 'mno']


@pytest.fixture
def vocab(words):
    """Vocabulary fixture."""
    return {word: idx for idx, word in enumerate(words)}


@pytest.fixture
def sequence_length():
    """Sequence length fixture."""
    return 2


@pytest.fixture
def dataset(words, vocab, sequence_length):
    """StockWordDataset fixture."""
    return StockWordDataset(
        words=words,
        vocab=vocab,
        sequence_length=sequence_length
    )


class TestStockWordDataset:
    """Test StockWordDataset class."""
    
    def test_dataset_initialization(self, dataset):
        """Test dataset initializes correctly."""
        assert dataset is not None
        assert len(dataset) == 3  # 5 words - 2 seq_length = 3 examples
    
    def test_dataset_length(self, words, vocab, sequence_length):
        """Test dataset returns correct length."""
        dataset = StockWordDataset(words, vocab, sequence_length)
        # With 5 words and sequence_length=2, we get 3 sequences
        expected_length = len(words) - sequence_length
        assert len(dataset) == expected_length
    
    def test_dataset_getitem_returns_tuple(self, dataset):
        """Test __getitem__ returns (input_ids, label) tuple."""
        input_ids, label = dataset[0]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(label, torch.Tensor)
    
    def test_dataset_getitem_shapes(self, dataset, sequence_length):
        """Test __getitem__ returns correct tensor shapes."""
        input_ids, label = dataset[0]
        assert input_ids.shape == (sequence_length,)
        assert label.shape == ()  # Scalar tensor
    
    def test_dataset_getitem_values(self, dataset, vocab):
        """Test __getitem__ returns correct token values."""
        # First sequence: words[0:2]=['abc', 'def'], label=words[2]='ghi'
        input_ids, label = dataset[0]
        assert input_ids[0].item() == vocab['abc']
        assert input_ids[1].item() == vocab['def']
        assert label.item() == vocab['ghi']
    
    def test_dataset_empty_words(self):
        """Test dataset with empty words list."""
        dataset = StockWordDataset([], {}, sequence_length=2)
        assert len(dataset) == 0
    
    def test_dataset_insufficient_words(self):
        """Test dataset when words < sequence_length."""
        words = ['abc', 'def']
        vocab = {w: i for i, w in enumerate(words)}
        dataset = StockWordDataset(words, vocab, sequence_length=5)
        assert len(dataset) == 0


class TestCustomCollateFn:
    """Test custom_collate_fn function."""
    
    @pytest.fixture
    def collate_dataset(self, words, vocab):
        """Dataset fixture for collate tests."""
        return StockWordDataset(words, vocab, sequence_length=2)
    
    def test_collate_fn_batch_stacking(self, collate_dataset):
        """Test collate_fn properly stacks batch."""
        batch = [collate_dataset[i] for i in range(2)]
        input_ids, labels = custom_collate_fn(batch)
        
        assert input_ids.shape[0] == 2  # batch size
        assert input_ids.shape[1] == 2  # sequence length
        assert labels.shape[0] == 2     # batch size
    
    def test_collate_fn_tensor_types(self, collate_dataset):
        """Test collate_fn returns correct tensor types."""
        batch = [collate_dataset[0]]
        input_ids, labels = custom_collate_fn(batch)
        
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long


class TestStockTransformerModel:
    """Test StockTransformerModel class."""
    
    @pytest.fixture
    def model(self):
        """Model fixture."""
        vocab_size = 7
        device = "cpu"
        return StockTransformerModel(
            vocab_size=vocab_size,
            max_position_embeddings=256,
            hidden_size=64,  # Smaller for testing
            num_hidden_layers=2,
            num_attention_heads=2,
            device=device
        )
    
    @pytest.fixture
    def vocab_size(self):
        """Vocab size fixture."""
        return 7
    
    def test_model_initialization(self, model, vocab_size):
        """Test model initializes correctly."""
        assert model is not None
        assert model.vocab_size == vocab_size
    
    def test_model_forward_pass(self, model, vocab_size):
        """Test forward pass with sample input."""
        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        outputs = model.forward(input_ids)
        assert 'logits' in outputs
        assert outputs['loss'] is None  # No labels provided
    
    def test_model_forward_with_labels(self, model, vocab_size):
        """Test forward pass with labels (computes loss)."""
        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        labels = torch.randint(0, vocab_size, (batch_size,))
        
        outputs = model.forward(input_ids, labels)
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['loss'] is not None
    
    def test_model_logits_shape(self, model, vocab_size):
        """Test logits output shape."""
        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        outputs = model.forward(input_ids)
        logits = outputs['logits']
        
        assert logits.shape[0] == batch_size
        assert logits.shape[1] == seq_length
        assert logits.shape[2] == vocab_size
    
    def test_model_predict(self, model, vocab_size):
        """Test prediction function."""
        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        predictions = model.predict(input_ids)
        
        assert predictions.shape[0] == batch_size
        assert (predictions >= 0).all()
        assert (predictions < vocab_size).all()
    
    def test_model_train_eval_modes(self, model):
        """Test switching between train and eval modes."""
        model.train()
        assert model.model.training
        
        model.eval()
        assert not model.model.training
    
    def test_model_parameters(self, model):
        """Test model has trainable parameters."""
        params = list(model.parameters())
        assert len(params) > 0
    
    def test_model_device_placement(self, model, vocab_size):
        """Test model is on correct device."""
        device = "cpu"
        input_ids = torch.randint(0, vocab_size, (1, 4)).to(device)
        outputs = model.forward(input_ids)
        
        logits = outputs['logits']
        assert logits.device.type == device


class TestDatasetPreconvertedTensors:
    """Test that dataset pre-converts tokens to tensors for performance."""
    
    @pytest.fixture
    def words(self):
        """Sample words fixture."""
        return ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr', 'stu', 'vwx']
    
    @pytest.fixture
    def vocab(self, words):
        """Vocabulary fixture."""
        return {word: idx for idx, word in enumerate(words)}
    
    def test_token_sequences_is_tensor(self, words, vocab):
        """Test that token_sequences is a pre-converted tensor."""
        dataset = StockWordDataset(words, vocab, sequence_length=2)
        assert isinstance(dataset.token_sequences, torch.Tensor)
        assert dataset.token_sequences.dtype == torch.long
    
    def test_getitem_returns_tensor_slices(self, words, vocab):
        """Test that __getitem__ returns slices of pre-converted tensor."""
        dataset = StockWordDataset(words, vocab, sequence_length=2)
        
        # Multiple calls should return consistent results
        input_ids_1, label_1 = dataset[0]
        input_ids_2, label_2 = dataset[0]
        
        assert torch.equal(input_ids_1, input_ids_2)
        assert torch.equal(label_1, label_2)
    
    def test_getitem_tensor_types(self, words, vocab):
        """Test that __getitem__ returns long tensors."""
        dataset = StockWordDataset(words, vocab, sequence_length=2)
        input_ids, label = dataset[0]
        
        assert input_ids.dtype == torch.long
        assert label.dtype == torch.long
    
    def test_empty_dataset_tensor(self):
        """Test that empty dataset creates empty tensor."""
        dataset = StockWordDataset([], {}, sequence_length=2)
        assert isinstance(dataset.token_sequences, torch.Tensor)
        assert len(dataset.token_sequences) == 0
    
    def test_dataset_tensor_values(self, words, vocab):
        """Test that tensor values match expected token IDs."""
        dataset = StockWordDataset(words, vocab, sequence_length=2)
        
        # Verify all tokens are correct
        for i, word in enumerate(words):
            expected_id = vocab[word]
            actual_id = dataset.token_sequences[i].item()
            assert actual_id == expected_id
