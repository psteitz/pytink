"""PyTorch dataset and model classes for stock prediction."""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for DataLoader to properly batch sequences and labels."""
    input_ids_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    
    # Stack input sequences: (batch_size, context_window_size)
    input_ids = torch.stack(input_ids_list)
    
    # Stack labels: (batch_size,)
    labels = torch.stack(labels_list)
    
    return input_ids, labels


class StockWordDataset(Dataset):
    """PyTorch dataset for stock word sequences."""
    
    def __init__(
        self,
        words: List[str],
        vocab: Dict[str, int],
        context_window_size: int = 4
    ):
        """
        Initialize dataset.
        
        Args:
            words: List of word strings
            vocab: Dictionary mapping words to token IDs
            context_window_size: Length of input sequences for prediction
        """
        self.words = words
        self.vocab = vocab
        self.context_window_size = context_window_size
        
        # Convert words to token sequences
        token_list = []
        for word in words:
            token_id = vocab.get(word)
            if token_id is not None:
                token_list.append(token_id)
        
        # Pre-convert to tensor for faster __getitem__ (avoids creating tensors on each call)
        self.token_sequences = torch.tensor(token_list, dtype=torch.long) if token_list else torch.tensor([], dtype=torch.long)
        
        logger.info(f"Created dataset with {len(self.token_sequences)} tokens")
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return max(0, len(self.token_sequences) - self.context_window_size)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return input and target sequences.
        
        For causal language modeling:
        - input_ids: Token sequence of length context_window_size
        - labels: Next token after the sequence (shape: scalar)
        
        Returns:
            input_ids: Token IDs for input sequence, shape (context_window_size,)
            labels: Token ID for next word, shape ()
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        
        # Slicing pre-converted tensor is much faster than creating new tensors
        input_ids = self.token_sequences[idx:idx + self.context_window_size]
        label = self.token_sequences[idx + self.context_window_size]
        
        return input_ids, label


class StockTransformerModel:
    """Wrapper for a transformer-based stock prediction model."""
    
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int = 256,
        hidden_size: int = 128,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        device: str = "cpu"
    ):
        """
        Initialize transformer model.
        
        Args:
            vocab_size: Number of unique words/tokens
            max_position_embeddings: Maximum sequence length
            hidden_size: Hidden dimension size
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            device: Device to use ('cpu' or 'cuda')
        """
        self.vocab_size = vocab_size
        self.device = device
        self.class_weights = None  # Will be set via set_class_weights()
        
        # Create custom configuration for GPT-2 style model
        config = AutoConfig.from_pretrained("gpt2")
        config.vocab_size = vocab_size
        config.max_position_embeddings = max_position_embeddings
        config.hidden_size = hidden_size
        config.num_hidden_layers = num_hidden_layers
        config.num_attention_heads = num_attention_heads
        
        # Initialize untrained model
        self.model = AutoModelForCausalLM.from_config(config)
        self.model.to(device)
        
        logger.info(
            f"Initialized transformer model with vocab_size={vocab_size}, "
            f"hidden_size={hidden_size}, num_layers={num_hidden_layers}"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            labels: Target token IDs of shape (batch_size,) for next-word prediction
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        input_ids = input_ids.to(self.device)
        
        # Get model outputs
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits  # shape: (batch_size, seq_length, vocab_size)
        
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            # Use only the last token's logits for next-token prediction
            # Shift logits: we want to predict the token after the sequence
            last_logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fn(last_logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get predictions for input sequences.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
        
        Returns:
            Predicted token IDs of shape (batch_size,)
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids)
            # Take the last token's logits and get argmax
            last_logits = outputs['logits'][:, -1, :]
            predictions = torch.argmax(last_logits, dim=-1)
        return predictions
    
    def get_model(self) -> nn.Module:
        """Return the underlying PyTorch model."""
        return self.model
    
    def parameters(self):
        """Return model parameters for optimizer."""
        return self.model.parameters()
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def set_class_weights(self, weights: torch.Tensor):
        """Set class weights for weighted cross-entropy loss.
        
        Args:
            weights: Tensor of shape (vocab_size,) with weight for each class.
                     Higher weights = more penalty for misclassifying that class.
        """
        self.class_weights = weights.to(self.device)
        logger.info(f"Set class weights (min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f})")
