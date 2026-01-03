#!/usr/bin/env python3
"""
Command-line script to run stock price prediction analysis.
Usage: python train_model.py --stocks 20 --interval 15 --epochs 10 --batch-size 64 --sequence-length 8
"""
import argparse
import sys
import logging
import time
import json
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
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from database import StockDatabase
from processor import PriceProcessor
from model import StockWordDataset, StockTransformerModel, custom_collate_fn


class BatchProgressFilter(logging.Filter):
    """Filter out batch progress messages (keep only epoch summaries)."""
    def filter(self, record):
        # Suppress messages containing "Batch" (e.g., "Epoch 6/10, Batch 60/28661, Loss: 2.4032")
        # Keep messages with epoch summaries (e.g., "Epoch 5/10 - Train Loss:")
        if "Batch" in record.getMessage():
            return False
        return True


# Setup logging to both console and file
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.addFilter(BatchProgressFilter())
logger.info(f"Log file: {log_file}")


def filter_stocks_by_recency(db, stock_ids, random_stocks, quotes_dict, max_age_days=30):
    """Filter out stocks with stale data (last quote older than max_age_days).
    
    Args:
        db: StockDatabase instance
        stock_ids: List of stock IDs to filter
        random_stocks: List of stock dicts with id, ticker, name
        quotes_dict: Dict mapping stock_id to list of quotes
        max_age_days: Maximum age of last quote in days (default 30)
    
    Returns:
        Tuple of (filtered_stock_ids, filtered_random_stocks, filtered_quotes_dict)
    """
    # Find the most recent quote timestamp across all stocks
    max_timestamp = None
    for stock_id in stock_ids:
        if stock_id in quotes_dict and quotes_dict[stock_id]:
            last_quote = quotes_dict[stock_id][-1]
            if isinstance(last_quote['timestamp'], str):
                ts = datetime.fromisoformat(last_quote['timestamp'])
            else:
                ts = last_quote['timestamp']
            if max_timestamp is None or ts > max_timestamp:
                max_timestamp = ts
    
    if max_timestamp is None:
        logger.warning("No quotes found in any stock")
        return stock_ids, random_stocks, quotes_dict
    
    cutoff_date = max_timestamp - timedelta(days=max_age_days)
    logger.info(f"Filtering stocks: keeping only those with quotes after {cutoff_date}")
    
    # Create lookup dicts for faster filtering
    stock_dict = {s['id']: s for s in random_stocks}
    
    # Filter out stale stocks
    valid_stock_ids = []
    valid_random_stocks = []
    valid_quotes_dict = {}
    removed_tickers = []
    
    for stock_id in stock_ids:
        if stock_id not in quotes_dict or not quotes_dict[stock_id]:
            logger.debug(f"  Removing stock {stock_id}: no quotes")
            continue
        
        last_quote = quotes_dict[stock_id][-1]
        if isinstance(last_quote['timestamp'], str):
            last_ts = datetime.fromisoformat(last_quote['timestamp'])
        else:
            last_ts = last_quote['timestamp']
        
        if last_ts < cutoff_date:
            ticker = stock_dict.get(stock_id, {}).get('ticker', 'Unknown')
            days_old = (max_timestamp - last_ts).days
            logger.info(f"  Removing stock {ticker}: last quote is {days_old} days old")
            removed_tickers.append(ticker)
            continue
        
        valid_stock_ids.append(stock_id)
        if stock_id in stock_dict:
            valid_random_stocks.append(stock_dict[stock_id])
        valid_quotes_dict[stock_id] = quotes_dict[stock_id]
    
    # If we removed stocks, fetch replacements
    removed_count = len(stock_ids) - len(valid_stock_ids)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} stale stocks ({', '.join(removed_tickers)}), fetching replacements...")
        replacement_stocks = db.get_random_stocks(count=removed_count, min_quotes=100000)
        
        # Remove duplicate stocks by ID
        seen_ids = set(valid_stock_ids)
        unique_replacements = []
        for stock in replacement_stocks:
            if stock['id'] not in seen_ids:
                unique_replacements.append(stock)
                seen_ids.add(stock['id'])
        
        if unique_replacements:
            replacement_ids = [s['id'] for s in unique_replacements]
            replacement_quotes = db.get_quotes_for_stocks(replacement_ids)
            
            # Recursively filter replacements (limited to 1 level to avoid excessive queries)
            replacement_ids, replacement_objs, replacement_q = filter_stocks_by_recency(
                db, replacement_ids, unique_replacements, replacement_quotes, max_age_days
            )
            
            valid_stock_ids.extend(replacement_ids)
            valid_random_stocks.extend(replacement_objs)
            valid_quotes_dict.update(replacement_q)
    
    return valid_stock_ids, valid_random_stocks, valid_quotes_dict


def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dictionary with configuration values
    """
    if yaml is None:
        logger.error("PyYAML not installed. Install with: pip install pyyaml")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return {}


def save_model(model, output_dir, logger, tickers=None, config=None, args=None, log_file=None, delta_values=None):
    """Save trained model, config, and log file to a dedicated subdirectory.
    
    Directory structure: output_dir/<tickers>/<timestamp>/
    E.g., models/AAPL-GOOGL-MSFT/20260101_143052/
    
    Args:
        model: The trained model to save
        output_dir: Base directory for models
        logger: Logger instance
        tickers: List of stock tickers to include in directory/filename
        config: Configuration dict used for training
        args: Argument namespace with training parameters
        log_file: Path to the log file to copy
        delta_values: List of delta values used for encoding
    """
    import shutil
    
    # Create subdirectory with tickers name
    if tickers:
        ticker_str = '-'.join(sorted(tickers))
    else:
        ticker_str = "model"
    
    # Create timestamp subdirectory for this run
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_dir = Path(output_dir) / ticker_str / timestamp_str
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = model_dir / "model.pt"
    try:
        torch.save(model.get_model().state_dict(), model_path)
        logger.info(f"✓ Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return
    
    # Save config file in same format as config_template.yaml
    # This config can be used directly with --config parameter
    config_path = model_dir / "config.yaml"
    try:
        # Build config in the same structure as config_template.yaml
        training_config = {
            'data': {
                'interval_minutes': args.interval if args else 15,
                'sequence_length': args.sequence_length if args else 8,
            },
            'training': {
                'batch_size': args.batch_size if args else 64,
                'num_epochs': args.epochs if args else 10,
                'learning_rate': args.learning_rate if args else 1e-5,
            },
            'output': {
                'save_model': True,
            },
        }
        
        # Add tickers list if available, otherwise num_stocks
        if tickers:
            training_config['data']['tickers'] = sorted(tickers)
        else:
            training_config['data']['num_stocks'] = args.stocks if args else 20
        
        # Add delta_ranges if custom values were used
        if delta_values is not None:
            training_config['delta_ranges'] = delta_values
        elif config and 'delta_ranges' in config:
            training_config['delta_ranges'] = config['delta_ranges']
        
        if yaml is not None:
            with open(config_path, 'w') as f:
                f.write("# Configuration used for this training run\n")
                f.write("# Can be used with: python train_model.py --db-password PASSWORD --config config.yaml\n\n")
                yaml.dump(training_config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"✓ Config saved to {config_path}")
        else:
            logger.warning("PyYAML not available, skipping config save")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
    
    # Copy log file
    if log_file and Path(log_file).exists():
        log_dest = model_dir / "training.log"
        try:
            shutil.copy(log_file, log_dest)
            logger.info(f"✓ Log file copied to {log_dest}")
        except Exception as e:
            logger.error(f"Failed to copy log file: {e}")


def save_vocabulary(vocab, output_dir, logger):
    """Save vocabulary mapping to JSON file."""
    output_path = Path(output_dir) / "vocabulary.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        logger.info(f"✓ Vocabulary saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save vocabulary: {e}")


def save_predictions(predictions, output_dir, logger):
    """Save predictions to file.
    
    Args:
        predictions: Dict with keys 'true_labels', 'pred_labels', 'sequences'
    """
    output_path = Path(output_dir) / "predictions.json"
    try:
        # Convert numpy arrays and tensors to lists for JSON serialization
        serializable = {
            'true_labels': [int(x) for x in predictions.get('true_labels', [])],
            'pred_labels': [int(x) for x in predictions.get('pred_labels', [])],
            'sequences': predictions.get('sequences', []),
            'accuracy': float(predictions.get('accuracy', 0))
        }
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"✓ Predictions saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")


def plot_results(training_history, eval_history, output_dir, logger):
    """Generate and save result plots.
    
    Args:
        training_history: Dict with 'epochs' and 'losses'
        eval_history: Dict with 'epochs', 'losses', 'accuracies'
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Plot training and eval loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        if 'epochs' in training_history and 'losses' in training_history:
            ax1.plot(training_history['epochs'], training_history['losses'], 'b-', label='Train Loss')
        if 'epochs' in eval_history and 'losses' in eval_history:
            ax1.plot(eval_history['epochs'], eval_history['losses'], 'r-', label='Eval Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Evaluation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'epochs' in eval_history and 'accuracies' in eval_history:
            ax2.plot(eval_history['epochs'], eval_history['accuracies'], 'g-', label='Eval Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Evaluation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "results.png"
        plt.savefig(plot_path, dpi=100)
        logger.info(f"✓ Results plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")


def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction Model')
    parser.add_argument('--db-password', type=str, required=True, help='Database password (required)')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--stocks', type=int, default=20, help='Number of random stocks to use')
    parser.add_argument('--interval', type=int, default=30, help='Time interval in minutes')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=16, help='Sequence length for context')
    parser.add_argument('--save-model', type=lambda x: x.lower() != 'false', default=True, 
                        help='Save trained model to disk (default: True)')
    
    args = parser.parse_args()
    
    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Merge config file values with command-line args (CLI args take precedence)
    if config:
        # Data config
        args.stocks = config.get('data', {}).get('num_stocks', args.stocks)
        args.interval = config.get('data', {}).get('interval_minutes', args.interval)
        args.sequence_length = config.get('data', {}).get('sequence_length', args.sequence_length)
        args.tickers = config.get('data', {}).get('tickers', None)
        
        # Training config
        args.batch_size = config.get('training', {}).get('batch_size', args.batch_size)
        args.epochs = config.get('training', {}).get('num_epochs', args.epochs)
        args.learning_rate = config.get('training', {}).get('learning_rate', args.learning_rate)
        
        # Output config - CLI --save-model takes precedence, then config, then default True
        args.save_model = config.get('output', {}).get('save_model', args.save_model)
        args.save_vocabulary = config.get('output', {}).get('save_vocabulary', False)
        args.save_predictions = config.get('output', {}).get('save_predictions', False)
        args.plot_results = config.get('output', {}).get('plot_results', False)
    else:
        # Set defaults for output options (save_model already set via CLI default)
        args.save_vocabulary = False
        args.save_predictions = False
        args.plot_results = False
        args.tickers = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Log run parameters
    logger.info("="*60)
    logger.info("RUN PARAMETERS")
    logger.info("="*60)
    logger.info(f"Number of stocks: {args.stocks}")
    logger.info(f"Context window sequence length: {args.sequence_length}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Number of epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Interval (minutes): {args.interval}")
    logger.info("="*60)
    
    # 1. Connect to database
    logger.info("Connecting to database...")
    db = StockDatabase(password=args.db_password)
    db.connect()
    
    # 2. Get stocks - either by tickers list or randomly
    if args.tickers:
        logger.info(f"Fetching specified tickers: {args.tickers}")
        random_stocks = db.get_stocks_by_tickers(args.tickers)
        
        # Check for missing tickers - exit with error if any not found
        found_tickers = {stock['ticker'] for stock in random_stocks}
        missing_tickers = [t for t in args.tickers if t not in found_tickers]
        if missing_tickers:
            logger.error(f"Tickers not found in database: {missing_tickers}")
            db.close()
            sys.exit(1)
    else:
        logger.info(f"Fetching {args.stocks} random stocks with at least 100,000 quotes...")
        random_stocks = db.get_random_stocks(count=args.stocks, min_quotes=100000)
    
    # Remove duplicate stocks by ID (keep first occurrence)
    seen_ids = set()
    unique_stocks = []
    for stock in random_stocks:
        if stock['id'] not in seen_ids:
            unique_stocks.append(stock)
            seen_ids.add(stock['id'])
    
    if len(unique_stocks) < len(random_stocks):
        logger.warning(f"Removed {len(random_stocks) - len(unique_stocks)} duplicate stocks")
    
    random_stocks = unique_stocks
    stock_ids = [stock['id'] for stock in random_stocks]
    
    if len(random_stocks) < args.stocks:
        logger.warning(f"Only found {len(random_stocks)} stocks with >= 100,000 quotes (requested {args.stocks})")
    
    # Update missing stock names from yFinance
    logger.info("Updating missing stock names...")
    db.update_missing_stock_names(stock_ids)
    
    # Refresh the stock data for the selected IDs to get updated names
    cursor = db.connection.cursor(dictionary=True)
    placeholders = ','.join(['%s'] * len(stock_ids))
    cursor.execute(f"SELECT id, ticker, name FROM stocks WHERE id IN ({placeholders})", stock_ids)
    random_stocks = cursor.fetchall()
    cursor.close()
    
    for stock in random_stocks:
        name = stock['name'] or 'N/A'
        logger.info(f"  {stock['ticker']}: {name}")
    
    # 3. Fetch quotes
    logger.info("Fetching quote data...")
    data_start_time = time.time()
    quotes_dict = db.get_quotes_for_stocks(stock_ids)
    
    for stock_id, quotes in quotes_dict.items():
        ticker = next((s['ticker'] for s in random_stocks if s['id'] == stock_id), 'Unknown')
        logger.info(f"  {ticker}: {len(quotes)} quotes")
    
    # Filter out stocks with stale data (last quote > 30 days old)
    logger.info("Filtering stocks by data recency (max 30 days old)...")
    stock_ids, random_stocks, quotes_dict = filter_stocks_by_recency(
        db, stock_ids, random_stocks, quotes_dict, max_age_days=30
    )
    
    logger.info(f"Using {len(stock_ids)} stocks after filtering:")
    for stock in random_stocks:
        ticker = stock['ticker']
        stock_id = stock['id']
        quote_count = len(quotes_dict.get(stock_id, []))
        name = stock['name'] or 'N/A'
        logger.info(f"  {ticker}: {quote_count} quotes - {name}")
    
    # 4. Process data
    logger.info(f"Processing data with {args.interval}-minute intervals...")
    
    # Get custom delta ranges from config if provided
    delta_values = None
    if config and 'delta_ranges' in config:
        delta_values = config['delta_ranges']
        logger.info(f"Using custom delta ranges: {delta_values}")
    
    processor = PriceProcessor(interval_minutes=args.interval, delta_values=delta_values)
    words = processor.extract_words(quotes_dict, stock_ids)
    
    logger.info(f"Generated {len(words)} words")
    
    # 5. Analyze vocabulary and delta distributions
    unique_count, unique_words = processor.count_unique_words(words)
    logger.info(f"Unique words: {unique_count}")
    logger.info(f"Vocabulary coverage: {unique_count / len(words) * 100:.2f}%")
    
    # Display top 10 most common price movement patterns
    word_freq = Counter(words)
    logger.info("Top 10 most common price movement patterns:")
    for word, count in word_freq.most_common(10):
        pct = count / len(words) * 100
        logger.info(f"  '{word}': {count:6} times ({pct:5.2f}%)")
    
    if len(words) == 0:
        logger.error("No words generated. Check data alignment and timestamp coverage.")
        return
    
    # Compute delta frequency distributions
    logger.info(f"Delta frequency distributions ({args.interval}-minute intervals):")
    delta_labels = ['a (-1%)', 'b (-0.5%)', 'c (-0.1%)', 'd (0%)', 'e (+0.1%)', 'f (+0.5%)', 'g (+1%)']
    delta_counts = {label: 0 for label in delta_labels}
    
    for word in words:
        for i, char in enumerate(word):
            if char.isalpha():
                delta_idx = ord(char) - ord('a')
                if 0 <= delta_idx < len(delta_labels):
                    delta_counts[delta_labels[delta_idx]] += 1
    
    total_deltas = sum(delta_counts.values())
    logger.info(f"Total deltas: {total_deltas}")
    logger.info("Distribution:")
    for label, count in delta_counts.items():
        pct = (count / total_deltas * 100) if total_deltas > 0 else 0
        bar_length = int(pct / 2)  # Scale to ~50 chars max
        bar = '█' * bar_length
        logger.info(f"  {label:15} {count:8} ({pct:6.2f}%) {bar}")
    
    # 6. Create dataset and vocab
    vocab = {word: idx for idx, word in enumerate(sorted(unique_words))}
    dataset = StockWordDataset(words=words, vocab=vocab, sequence_length=args.sequence_length)
    
    logger.info(f"Created dataset with {len(dataset)} sequences")
    
    if len(dataset) == 0:
        logger.error("Dataset is empty. Need more data or shorter sequence length.")
        return
    
    # 7. Split dataset into train/eval
    # Use last 15% of sequences for evaluation (maintains temporal ordering)
    eval_split = int(len(dataset) * 0.85)
    train_dataset = torch.utils.data.Subset(dataset, range(0, eval_split))
    eval_dataset = torch.utils.data.Subset(dataset, range(eval_split, len(dataset)))
    
    logger.info(f"Train sequences: {len(train_dataset)}, Eval sequences: {len(eval_dataset)}")
    
    # 8. Initialize model (Stage 1: larger model)
    data_elapsed = time.time() - data_start_time
    logger.info(f"Data preparation completed in {data_elapsed:.2f} seconds ({data_elapsed/60:.2f} minutes)")
    
    model_start_time = time.time()
    model = StockTransformerModel(
        vocab_size=len(vocab),
        max_position_embeddings=256,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        device=device
    )
    
    # 9. Training
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    best_eval_loss = float('inf')
    
    model.train()
    for epoch in range(args.epochs):
        # Training phase
        epoch_loss = 0.0
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Evaluation phase
        model.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
        
        with torch.no_grad():
            for input_ids, labels in eval_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids=input_ids, labels=labels)
                eval_loss += outputs['loss'].item() * input_ids.size(0)
                
                logits = outputs['logits']
                predictions = torch.argmax(logits[:, -1, :], dim=-1)
                eval_correct += (predictions == labels).sum().item()
                eval_total += labels.size(0)
        
        avg_eval_loss = eval_loss / len(eval_dataset)
        eval_accuracy = eval_correct / eval_total
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Eval Loss: {avg_eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            logger.info(f"  ✓ Best eval loss improved to {best_eval_loss:.4f}")
        
        model.train()
    
    # Final evaluation on eval set
    training_elapsed = time.time() - model_start_time
    logger.info(f"Model training completed in {training_elapsed:.2f} seconds ({training_elapsed/60:.2f} minutes)")
    
    logger.info("="*60)
    logger.info("FINAL EVALUATION ON HELD-OUT EVAL SET")
    logger.info("="*60)
    
    model.eval()
    eval_loss = 0.0
    eval_correct = 0
    eval_total = 0
    
    with torch.no_grad():
        for input_ids, labels in eval_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids=input_ids, labels=labels)
            eval_loss += outputs['loss'].item() * input_ids.size(0)
            
            logits = outputs['logits']
            predictions = torch.argmax(logits[:, -1, :], dim=-1)
            eval_correct += (predictions == labels).sum().item()
            eval_total += labels.size(0)
    
    final_eval_loss = eval_loss / len(eval_dataset)
    final_eval_accuracy = eval_correct / eval_total
    final_perplexity = np.exp(final_eval_loss)
    
    logger.info(f"Final Eval Loss: {final_eval_loss:.4f}")
    logger.info(f"Final Eval Accuracy: {final_eval_accuracy:.4f}")
    logger.info(f"Final Perplexity: {final_perplexity:.4f}")
    logger.info("="*60)
    
    # Get tickers for confusion matrix and model filename
    tickers = [stock['ticker'] for stock in random_stocks]
    
    # Generate per-stock confusion matrices
    logger.info("")
    logger.info("="*60)
    logger.info("PER-STOCK CONFUSION MATRICES (Letter-by-Letter Analysis)")
    logger.info("="*60)
    
    # Collect predicted and actual words
    all_true_indices = []
    all_pred_indices = []
    with torch.no_grad():
        for input_ids, labels in eval_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids=input_ids, labels=labels)
            logits = outputs['logits']
            predictions = torch.argmax(logits[:, -1, :], dim=-1)
            all_true_indices.extend(labels.cpu().numpy())
            all_pred_indices.extend(predictions.cpu().numpy())
    
    # Create reverse vocab mapping (index -> word)
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # Convert indices to words
    true_words = [idx_to_word.get(idx, '?'*len(tickers)) for idx in all_true_indices]
    pred_words = [idx_to_word.get(idx, '?'*len(tickers)) for idx in all_pred_indices]
    
    # Get the letters used (typically a-g)
    delta_letters = [chr(ord('a') + i) for i in range(len(processor.delta_values))]
    
    # For each stock position, build confusion matrix
    num_stocks = len(tickers)
    for stock_idx, ticker in enumerate(tickers):
        # Collect actual and predicted letters for this stock position
        actual_letters = []
        predicted_letters = []
        
        for true_word, pred_word in zip(true_words, pred_words):
            if stock_idx < len(true_word) and stock_idx < len(pred_word):
                actual_letters.append(true_word[stock_idx])
                predicted_letters.append(pred_word[stock_idx])
        
        if not actual_letters:
            continue
        
        # Build confusion matrix
        # Rows = actual, Columns = predicted
        confusion = {actual: {pred: 0 for pred in delta_letters} for actual in delta_letters}
        
        for actual, predicted in zip(actual_letters, predicted_letters):
            if actual in confusion and predicted in delta_letters:
                confusion[actual][predicted] += 1
        
        # Calculate per-stock accuracy
        correct = sum(1 for a, p in zip(actual_letters, predicted_letters) if a == p)
        stock_accuracy = correct / len(actual_letters) if actual_letters else 0
        
        logger.info(f"\n{ticker} (position {stock_idx + 1}/{num_stocks}) - Accuracy: {stock_accuracy:.4f}")
        
        # Print confusion matrix header
        header = "Actual\\Pred | " + " | ".join(f" {l} " for l in delta_letters) + " | Total"
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))
        
        # Print each row
        for actual in delta_letters:
            row_total = sum(confusion[actual].values())
            if row_total > 0:  # Only show rows with data
                row_values = " | ".join(f"{confusion[actual][pred]:3d}" for pred in delta_letters)
                logger.info(f"     {actual}      | {row_values} | {row_total:5d}")
        
        logger.info("-" * len(header))
    
    # Summary statistics across all stocks
    logger.info("\n" + "="*60)
    logger.info("SUMMARY: Per-Stock Accuracies")
    logger.info("="*60)
    
    stock_accuracies = []
    for stock_idx, ticker in enumerate(tickers):
        actual_letters = [true_words[i][stock_idx] for i in range(len(true_words)) 
                         if stock_idx < len(true_words[i]) and stock_idx < len(pred_words[i])]
        predicted_letters = [pred_words[i][stock_idx] for i in range(len(pred_words))
                            if stock_idx < len(true_words[i]) and stock_idx < len(pred_words[i])]
        
        if actual_letters:
            correct = sum(1 for a, p in zip(actual_letters, predicted_letters) if a == p)
            acc = correct / len(actual_letters)
            stock_accuracies.append((ticker, acc))
    
    # Sort by accuracy
    stock_accuracies.sort(key=lambda x: x[1], reverse=True)
    for ticker, acc in stock_accuracies:
        logger.info(f"  {ticker:6s}: {acc:.4f}")
    
    if stock_accuracies:
        avg_stock_acc = sum(acc for _, acc in stock_accuracies) / len(stock_accuracies)
        logger.info(f"\n  Average per-stock accuracy: {avg_stock_acc:.4f}")
    
    logger.info("="*60)
    
    logger.info("Analysis complete!")
    
    # Create output directory for saved artifacts
    output_dir = log_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create models directory for saved models (at same level as logs)
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Save artifacts if requested
    if args.save_model:
        save_model(model, models_dir, logger, tickers=tickers, config=config, args=args, log_file=log_file, delta_values=delta_values)
    
    if args.save_vocabulary:
        save_vocabulary(vocab, output_dir, logger)
    
    if args.save_predictions:
        # Collect predictions from final eval
        all_true = []
        all_pred = []
        with torch.no_grad():
            for input_ids, labels in eval_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids=input_ids, labels=labels)
                logits = outputs['logits']
                predictions = torch.argmax(logits[:, -1, :], dim=-1)
                all_true.extend(labels.cpu().numpy())
                all_pred.extend(predictions.cpu().numpy())
        
        predictions_dict = {
            'true_labels': all_true,
            'pred_labels': all_pred,
            'accuracy': final_eval_accuracy
        }
        save_predictions(predictions_dict, output_dir, logger)
    
    if args.plot_results:
        training_history_dict = {
            'epochs': list(range(1, args.epochs + 1)),
            'losses': []  # Would need to track this during training
        }
        eval_history_dict = {
            'epochs': list(range(1, args.epochs + 1)),
            'losses': [],  # Would need to track this during training
            'accuracies': []  # Would need to track this during training
        }
        plot_results(training_history_dict, eval_history_dict, output_dir, logger)
    db.close()


if __name__ == '__main__':
    import numpy as np
    main()
