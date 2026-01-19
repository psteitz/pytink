#!/usr/bin/env python3
"""
Inference script to evaluate a trained model on recent data.

Loads a model from a directory and evaluates its performance on data from
the last 3 months (relative to the most recent date where all stocks have quotes).

Usage: python inference.py --db-password PASSWORD --model-dir models/TICKER-LIST/TIMESTAMP/
"""
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

try:
    import yaml
except ImportError:
    yaml = None

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from torch.utils.data import DataLoader
import numpy as np

from database import StockDatabase
from processor import PriceProcessor
from model import StockWordDataset, StockTransformerModel, custom_collate_fn


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_model_config(model_dir: Path) -> dict:
    """Load configuration from model directory.
    
    Args:
        model_dir: Path to model directory containing config.yaml
        
    Returns:
        Configuration dictionary
    """
    config_path = model_dir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if yaml is None:
        raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_model(model_dir: Path, vocab_size: int, config: dict, device: str) -> StockTransformerModel:
    """Load trained model from directory.
    
    Args:
        model_dir: Path to model directory containing model.pt
        vocab_size: Size of vocabulary
        config: Model configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded StockTransformerModel
    """
    model_path = model_dir / 'model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_config = config.get('model', {})
    model = StockTransformerModel(
        vocab_size=vocab_size,
        max_position_embeddings=model_config.get('max_position_embeddings', 256),
        hidden_size=model_config.get('hidden_size', 128),
        num_hidden_layers=model_config.get('num_hidden_layers', 4),
        num_attention_heads=model_config.get('num_attention_heads', 4),
        device=device
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.get_model().load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model


def find_common_date_range(quotes_dict: dict, stock_ids: list) -> tuple:
    """Find the date range where all stocks have data.
    
    Args:
        quotes_dict: Dictionary mapping stock_id to list of quotes
        stock_ids: List of stock IDs to consider
        
    Returns:
        Tuple of (min_date, max_date) where all stocks have quotes
    """
    # Find the latest start date and earliest end date across all stocks
    latest_start = None
    earliest_end = None
    
    for stock_id in stock_ids:
        quotes = quotes_dict.get(stock_id, [])
        if not quotes:
            continue
            
        # Get first and last quote timestamps
        first_ts = quotes[0]['timestamp']
        last_ts = quotes[-1]['timestamp']
        
        if isinstance(first_ts, str):
            first_ts = datetime.fromisoformat(first_ts)
        if isinstance(last_ts, str):
            last_ts = datetime.fromisoformat(last_ts)
        
        if latest_start is None or first_ts > latest_start:
            latest_start = first_ts
        if earliest_end is None or last_ts < earliest_end:
            earliest_end = last_ts
    
    return latest_start, earliest_end


def filter_quotes_by_date(quotes_dict: dict, start_date: datetime, end_date: datetime) -> dict:
    """Filter quotes to only include those within date range.
    
    Args:
        quotes_dict: Dictionary mapping stock_id to list of quotes
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        
    Returns:
        Filtered quotes dictionary
    """
    filtered = {}
    for stock_id, quotes in quotes_dict.items():
        filtered_quotes = []
        for quote in quotes:
            ts = quote['timestamp']
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if start_date <= ts <= end_date:
                filtered_quotes.append(quote)
        filtered[stock_id] = filtered_quotes
    return filtered


def evaluate_model(model, data_loader, vocab, device, tickers, delta_values):
    """Evaluate model and generate metrics.
    
    Args:
        model: Trained StockTransformerModel
        data_loader: DataLoader with evaluation data
        vocab: Vocabulary dictionary (word -> index)
        device: Device for computation
        tickers: List of ticker symbols
        delta_values: List of delta threshold values
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_true_indices = []
    all_pred_indices = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            outputs = model.forward(input_ids=input_ids, labels=labels)
            logits = outputs['logits']
            predictions = torch.argmax(logits[:, -1, :], dim=-1)
            
            if outputs['loss'] is not None:
                total_loss += outputs['loss'].item() * input_ids.size(0)
                total_samples += input_ids.size(0)
            
            all_true_indices.extend(labels.cpu().numpy())
            all_pred_indices.extend(predictions.cpu().numpy())
    
    # Calculate overall metrics
    correct = sum(1 for t, p in zip(all_true_indices, all_pred_indices) if t == p)
    accuracy = correct / len(all_true_indices) if all_true_indices else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    perplexity = np.exp(avg_loss) if avg_loss > 0 else float('inf')
    
    # Create reverse vocab mapping
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # Convert indices to words
    true_words = [idx_to_word.get(idx, '?' * len(tickers)) for idx in all_true_indices]
    pred_words = [idx_to_word.get(idx, '?' * len(tickers)) for idx in all_pred_indices]
    
    # Get delta letters
    delta_letters = [chr(ord('a') + i) for i in range(len(delta_values))]
    
    # Per-stock metrics
    stock_metrics = {}
    for stock_idx, ticker in enumerate(tickers):
        actual_letters = []
        predicted_letters = []
        
        for true_word, pred_word in zip(true_words, pred_words):
            if stock_idx < len(true_word) and stock_idx < len(pred_word):
                actual_letters.append(true_word[stock_idx])
                predicted_letters.append(pred_word[stock_idx])
        
        if not actual_letters:
            continue
        
        # Build confusion matrix
        confusion = {actual: {pred: 0 for pred in delta_letters} for actual in delta_letters}
        for actual, predicted in zip(actual_letters, predicted_letters):
            if actual in confusion and predicted in delta_letters:
                confusion[actual][predicted] += 1
        
        # Calculate accuracy
        correct = sum(1 for a, p in zip(actual_letters, predicted_letters) if a == p)
        stock_accuracy = correct / len(actual_letters) if actual_letters else 0
        
        stock_metrics[ticker] = {
            'accuracy': stock_accuracy,
            'confusion': confusion,
            'actual_distribution': dict(Counter(actual_letters)),
            'predicted_distribution': dict(Counter(predicted_letters)),
            'total_samples': len(actual_letters)
        }
    
    return {
        'overall_accuracy': accuracy,
        'overall_loss': avg_loss,
        'perplexity': perplexity,
        'total_samples': len(all_true_indices),
        'stock_metrics': stock_metrics,
        'delta_letters': delta_letters
    }


def print_results(metrics: dict, tickers: list):
    """Print evaluation results.
    
    Args:
        metrics: Dictionary with evaluation metrics
        tickers: List of ticker symbols
    """
    logger.info("=" * 60)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples evaluated: {metrics['total_samples']}")
    logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    logger.info(f"Overall Loss: {metrics['overall_loss']:.4f}")
    logger.info(f"Perplexity: {metrics['perplexity']:.4f}")
    logger.info("=" * 60)
    
    logger.info("\nPER-STOCK CONFUSION MATRICES")
    logger.info("=" * 60)
    
    delta_letters = metrics['delta_letters']
    
    for ticker in tickers:
        if ticker not in metrics['stock_metrics']:
            continue
        
        stock_data = metrics['stock_metrics'][ticker]
        confusion = stock_data['confusion']
        
        logger.info(f"\n{ticker} - Accuracy: {stock_data['accuracy']:.4f} ({stock_data['total_samples']} samples)")
        logger.info(f"  Actual distribution: {stock_data['actual_distribution']}")
        
        # Print confusion matrix header
        header = "Actual\\Pred | " + " | ".join(f" {l} " for l in delta_letters) + " | Total"
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))
        
        # Print each row
        for actual in delta_letters:
            row_total = sum(confusion[actual].values())
            if row_total > 0:
                row_values = " | ".join(f"{confusion[actual][pred]:3d}" for pred in delta_letters)
                logger.info(f"     {actual}      | {row_values} | {row_total:5d}")
        
        logger.info("-" * len(header))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Per-Stock Accuracies")
    logger.info("=" * 60)
    
    accuracies = []
    for ticker in tickers:
        if ticker in metrics['stock_metrics']:
            acc = metrics['stock_metrics'][ticker]['accuracy']
            accuracies.append(acc)
            logger.info(f"  {ticker}: {acc:.4f}")
    
    if accuracies:
        logger.info(f"\nMean accuracy: {np.mean(accuracies):.4f}")
        logger.info(f"Std accuracy: {np.std(accuracies):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained stock prediction model')
    parser.add_argument('--db-password', type=str, required=True, help='Database password')
    parser.add_argument('--model-dir', type=str, required=True, 
                        help='Path to model directory (e.g., models/AAPL-GOOGL/20260101_120000/)')
    parser.add_argument('--months', type=int, default=3, 
                        help='Number of months of recent data to evaluate (default: 3)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Resolve model directory path
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).parent / model_dir
    
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        sys.exit(1)
    
    logger.info(f"Loading model from: {model_dir}")
    
    # Load configuration
    config = load_model_config(model_dir)
    
    # Extract parameters from config
    data_config = config.get('data', {})
    tickers = data_config.get('tickers', [])
    interval_minutes = data_config.get('interval_minutes', 30)
    context_window_size = data_config.get('context_window_size', 32)
    delta_values = config.get('delta_ranges', None)
    
    if not tickers:
        logger.error("No tickers found in model config")
        sys.exit(1)
    
    logger.info(f"Model tickers: {tickers}")
    logger.info(f"Interval: {interval_minutes} minutes")
    logger.info(f"Sequence length: {context_window_size}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Connect to database
    logger.info("Connecting to database...")
    db = StockDatabase(password=args.db_password)
    db.connect()
    
    try:
        # Get stocks by tickers
        stocks = db.get_stocks_by_tickers(tickers)
        found_tickers = {s['ticker'] for s in stocks}
        missing = [t for t in tickers if t not in found_tickers]
        if missing:
            logger.error(f"Tickers not found in database: {missing}")
            sys.exit(1)
        
        # Maintain order from config
        stocks_ordered = []
        for ticker in tickers:
            for s in stocks:
                if s['ticker'] == ticker:
                    stocks_ordered.append(s)
                    break
        stocks = stocks_ordered
        stock_ids = [s['id'] for s in stocks]
        
        logger.info(f"Found {len(stocks)} stocks in database")
        
        # Fetch all quotes
        logger.info("Fetching quote data...")
        quotes_dict = db.get_quotes_for_stocks(stock_ids)
        
        for stock_id, quotes in quotes_dict.items():
            ticker = next((s['ticker'] for s in stocks if s['id'] == stock_id), 'Unknown')
            logger.info(f"  {ticker}: {len(quotes)} quotes")
        
        # Find common date range
        _, max_common_date = find_common_date_range(quotes_dict, stock_ids)
        
        if max_common_date is None:
            logger.error("Could not determine common date range for stocks")
            sys.exit(1)
        
        # Calculate start date (N months before max common date)
        eval_start_date = max_common_date - timedelta(days=args.months * 30)
        
        logger.info(f"Evaluation period: {eval_start_date} to {max_common_date}")
        
        # Filter quotes to evaluation period
        eval_quotes = filter_quotes_by_date(quotes_dict, eval_start_date, max_common_date)
        
        for stock_id, quotes in eval_quotes.items():
            ticker = next((s['ticker'] for s in stocks if s['id'] == stock_id), 'Unknown')
            logger.info(f"  {ticker}: {len(quotes)} quotes in evaluation period")
        
        # Process data
        logger.info(f"Processing data with {interval_minutes}-minute intervals...")
        processor = PriceProcessor(interval_minutes=interval_minutes, delta_values=delta_values)
        words = processor.extract_words(eval_quotes, stock_ids)
        
        logger.info(f"Generated {len(words)} words")
        
        if len(words) == 0:
            logger.error("No words generated from evaluation data")
            sys.exit(1)
        
        # Build vocabulary from words
        unique_count, unique_words = processor.count_unique_words(words)
        vocab = {word: idx for idx, word in enumerate(sorted(unique_words))}
        
        logger.info(f"Vocabulary size: {len(vocab)}")
        
        # Create dataset
        dataset = StockWordDataset(words=words, vocab=vocab, context_window_size=context_window_size)
        logger.info(f"Created dataset with {len(dataset)} sequences")
        
        if len(dataset) == 0:
            logger.error("Dataset is empty")
            sys.exit(1)
        
        # Load model
        model = load_model(model_dir, len(vocab), config, device)
        
        # Create data loader
        data_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=custom_collate_fn
        )
        
        # Evaluate
        logger.info("Evaluating model...")
        metrics = evaluate_model(
            model, 
            data_loader, 
            vocab, 
            device, 
            tickers,
            processor.delta_values
        )
        
        # Print results
        print_results(metrics, tickers)
        
    finally:
        db.close()


if __name__ == '__main__':
    main()
