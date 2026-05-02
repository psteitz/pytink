"""Utilities for analyzing and visualizing training results."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import json


def plot_training_loss(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training loss over time.
    
    Args:
        history: Dictionary with 'epoch', 'batch', 'loss' keys
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss over batches
    plt.plot(history['loss'], linewidth=0.5, alpha=0.7)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Batches')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_epoch_loss(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot average loss per epoch.
    
    Args:
        history: Dictionary with 'epoch', 'loss' keys
        save_path: Optional path to save the figure
    """
    # Group by epoch
    epochs_data = {}
    for epoch, loss in zip(history['epoch'], history['loss']):
        if epoch not in epochs_data:
            epochs_data[epoch] = []
        epochs_data[epoch].append(loss)
    
    epoch_nums = sorted(epochs_data.keys())
    epoch_losses = [np.mean(epochs_data[e]) for e in epoch_nums]
    
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_nums, epoch_losses, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Per Epoch')
    plt.grid(True, alpha=0.3)
    plt.xticks(epoch_nums)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_word_frequency(word_freq: Dict[str, int], top_n: int = 20, save_path: str = None):
    """
    Plot most frequent words.
    
    Args:
        word_freq: Dictionary mapping words to frequencies
        top_n: Number of top words to display
        save_path: Optional path to save the figure
    """
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words = [w[0] for w in sorted_words]
    freqs = [w[1] for w in sorted_words]
    
    plt.figure(figsize=(12, 6))
    plt.barh(words, freqs)
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def analyze_prediction_quality(predictions: Dict, threshold: float = 0.5):
    """
    Analyze model prediction quality.
    
    Args:
        predictions: Dictionary with 'input', 'true', 'pred', 'confidence' keys
        threshold: Confidence threshold for evaluation
    """
    total = len(predictions['true'])
    correct = sum(1 for t, p in zip(predictions['true'], predictions['pred']) if t == p)
    accuracy = correct / total if total > 0 else 0
    
    high_conf = sum(1 for c in predictions['confidence'] if c >= threshold)
    high_conf_correct = sum(
        1 for c, t, p in zip(predictions['confidence'], predictions['true'], predictions['pred'])
        if c >= threshold and t == p
    )
    high_conf_accuracy = high_conf_correct / high_conf if high_conf > 0 else 0
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"High Confidence (>= {threshold}) Predictions: {high_conf}/{total}")
    print(f"High Confidence Accuracy: {high_conf_accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'high_conf_predictions': high_conf,
        'high_conf_accuracy': high_conf_accuracy
    }


def save_vocabulary(vocab: Dict[str, int], filepath: str):
    """Save vocabulary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary saved to {filepath}")


def load_vocabulary(filepath: str) -> Dict[str, int]:
    """Load vocabulary from JSON file."""
    with open(filepath, 'r') as f:
        vocab = json.load(f)
    return vocab
