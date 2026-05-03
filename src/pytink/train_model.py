#!/usr/bin/env python3
"""
Command-line script to run stock price prediction analysis.

Exported classes:
    BatchProgressFilter  -- logging.Filter that suppresses per-batch progress
                            messages, keeping only epoch-level summaries.
    PreparedData         -- namedtuple returned by prepare_data containing
                            DataLoaders, vocabulary, and processed word sequences.

Usage: python train_model.py --num-stocks 20 --interval 15 --epochs 10 --batch-size 64 --context-window-size 8
"""
import argparse
import sys
import logging
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, namedtuple

try:
    import yaml
except ImportError:
    yaml = None

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pytink.database import StockDatabase
from pytink.processor import PriceProcessor
from pytink.model import StockWordDataset, StockTransformerModel, custom_collate_fn


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


def load_default_config():
    """Load default configuration from config_template.yaml.
    
    Returns:
        Dictionary with default configuration values
    """
    template_path = Path(__file__).parent / 'config_template.yaml'
    if template_path.exists():
        return load_config(template_path)
    else:
        logger.warning(f"Default config template not found at {template_path}")
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
                'context_window_size': args.context_window_size if args else 8,
            },
            'model': {
                'hidden_size': args.hidden_size if args else 128,
                'num_hidden_layers': args.num_hidden_layers if args else 4,
                'num_attention_heads': args.num_attention_heads if args else 4,
                'max_position_embeddings': args.max_position_embeddings if args else 256,
            },
            'training': {
                'batch_size': args.batch_size if args else 64,
                'num_epochs': args.epochs if args else 25,
                'learning_rate': args.learning_rate if args else 1e-5,
                'weight_decay': args.weight_decay if args else 0.0,
                'early_stopping_patience': args.early_stopping_patience if args else 5,
            },
            'output': {
                'save_model': True,
            },
        }
        
        # Add tickers list if available, otherwise num_stocks
        if tickers:
            training_config['data']['tickers'] = sorted(tickers)
        else:
            training_config['data']['num_stocks'] = args.num_stocks if args else 20
        
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
    """Save vocabulary mapping to JSON file.

    Args:
        vocab: Dictionary mapping word strings to token IDs.
        output_dir: Directory in which to write ``vocabulary.json``.
        logger: Logger instance for status messages.
    """
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


def parse_date(val, name):
    """Parse a YYYY-MM-DD string into a datetime, or return None if val is None.

    Calls sys.exit(1) on invalid format.
    """
    if val is None:
        return None
    try:
        return datetime.strptime(val, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid {name} format '{val}' — expected YYYY-MM-DD")
        sys.exit(1)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Stock Price Prediction Model')
    parser.add_argument('--db-password', type=str, required=True, help='Database password (required)')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--num-stocks', type=int, default=None, help='Number of random stocks to use')
    parser.add_argument('--tickers', type=str, default=None,
                        help='JSON list of ticker symbols, e.g. \'["BAC", "AXP", "MSFT"]\' (overrides --num-stocks)')
    parser.add_argument('--interval', type=int, default=None, help='Time interval in minutes')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=None, help='Early stopping patience (0 to disable)')
    parser.add_argument('--batch-size', type=int, default=None, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay for regularization')
    parser.add_argument('--context-window-size', type=int, default=None, help='Context window size (number of tokens for model input)')
    parser.add_argument('--start-date', type=str, default=None, help='Earliest quote date to include in training (YYYY-MM-DD); default includes all historical data')
    parser.add_argument('--end-date', type=str, default=None, help='Latest quote date to include in training (YYYY-MM-DD); default includes up to the most recent data')
    parser.add_argument('--save-model', type=lambda x: x.lower() != 'false', default=None,
                        help='Save trained model to disk (default: True)')
    return parser


def _merge_config(args, default_config, user_config):
    """Resolve all config values with priority: CLI args > user config > default config > hardcoded fallback."""
    default_data = default_config.get('data', {})
    default_training = default_config.get('training', {})
    default_output = default_config.get('output', {})
    default_model = default_config.get('model', {})
    user_model = user_config.get('model', {})

    def cv(cli_val, section, key, fallback):
        """Return first non-None value from: CLI, user config, default config, fallback."""
        if cli_val is not None:
            return cli_val
        v = user_config.get(section, {}).get(key)
        if v is not None:
            return v
        v = default_config.get(section, {}).get(key)
        if v is not None:
            return v
        return fallback

    # Data
    args.num_stocks = cv(args.num_stocks, 'data', 'num_stocks', 10)
    args.interval = cv(args.interval, 'data', 'interval_minutes', 30)
    args.context_window_size = cv(args.context_window_size, 'data', 'context_window_size', 32)
    args.start_date = parse_date(cv(args.start_date, 'data', 'start_date', None), '--start-date')
    args.end_date = parse_date(cv(args.end_date, 'data', 'end_date', None), '--end-date')

    # Tickers (JSON string from CLI or list from config)
    if args.tickers is not None:
        try:
            args.tickers = json.loads(args.tickers)
            if not isinstance(args.tickers, list):
                logger.error("--tickers must be a JSON list, e.g. '[\"BAC\", \"AXP\"]'")
                sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse --tickers as JSON: {e}")
            sys.exit(1)
    else:
        args.tickers = user_config.get('data', {}).get('tickers', default_data.get('tickers', None))
    if args.tickers is not None and len(args.tickers) == 0:
        args.tickers = None

    # Training
    args.batch_size = cv(args.batch_size, 'training', 'batch_size', 64)
    args.epochs = cv(args.epochs, 'training', 'num_epochs', 25)
    args.learning_rate = cv(args.learning_rate, 'training', 'learning_rate', 0.0003)
    args.weight_decay = cv(args.weight_decay, 'training', 'weight_decay', 0.01)
    args.early_stopping_patience = cv(args.early_stopping_patience, 'training', 'early_stopping_patience', 5)
    args.use_class_weights = user_config.get('training', {}).get('use_class_weights', default_training.get('use_class_weights', True))

    # Output
    args.save_model = cv(args.save_model, 'output', 'save_model', True)
    args.save_vocabulary = user_config.get('output', {}).get('save_vocabulary', default_output.get('save_vocabulary', False))
    args.save_predictions = user_config.get('output', {}).get('save_predictions', default_output.get('save_predictions', False))
    args.plot_results = user_config.get('output', {}).get('plot_results', default_output.get('plot_results', False))

    # Model architecture
    args.hidden_size = user_model.get('hidden_size', default_model.get('hidden_size', 128))
    args.num_hidden_layers = user_model.get('num_hidden_layers', default_model.get('num_hidden_layers', 4))
    args.num_attention_heads = user_model.get('num_attention_heads', default_model.get('num_attention_heads', 4))
    args.max_position_embeddings = user_model.get('max_position_embeddings', default_model.get('max_position_embeddings', 256))

    # Delta ranges
    args.delta_ranges = user_config.get('delta_ranges', default_config.get('delta_ranges', None))


def _fetch_stocks(db, args):
    """Return (stock_ids, random_stocks) from the database, deduped."""
    if args.tickers:
        logger.info(f"Fetching specified tickers: {args.tickers}")
        stocks = db.get_stocks_by_tickers(args.tickers)
        found = {s['ticker'] for s in stocks}
        missing = [t for t in args.tickers if t not in found]
        if missing:
            logger.error(f"Tickers not found in database: {missing}")
            db.close()
            sys.exit(1)
    else:
        logger.info(f"Fetching {args.num_stocks} random stocks with at least 100,000 quotes...")
        stocks = db.get_random_stocks(count=args.num_stocks, min_quotes=100000)

    # Dedupe by ID
    seen, unique = set(), []
    for s in stocks:
        if s['id'] not in seen:
            unique.append(s)
            seen.add(s['id'])
    if len(unique) < len(stocks):
        logger.warning(f"Removed {len(stocks) - len(unique)} duplicate stocks")
    if not args.tickers and len(unique) < args.num_stocks:
        logger.warning(f"Only found {len(unique)} stocks with >= 100,000 quotes (requested {args.num_stocks})")

    return [s['id'] for s in unique], unique


def _refresh_stock_names(db, stock_ids):
    """Update missing names via yFinance and return refreshed stock rows."""
    logger.info("Updating missing stock names...")
    db.update_missing_stock_names(stock_ids)
    cursor = db.connection.cursor(dictionary=True)
    placeholders = ','.join(['%s'] * len(stock_ids))
    cursor.execute(f"SELECT id, ticker, name FROM stocks WHERE id IN ({placeholders})", stock_ids)
    stocks = cursor.fetchall()
    cursor.close()
    for s in stocks:
        logger.info(f"  {s['ticker']}: {s['name'] or 'N/A'}")
    return stocks


def _compute_class_weights(train_dataset, vocab_size, device):
    """Return a log-scaled, normalised class-weight tensor."""
    class_counts = torch.zeros(vocab_size)
    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        class_counts[label.item()] += 1

    total = class_counts.sum()
    weights = torch.zeros(vocab_size)
    mask = class_counts > 0
    freqs = class_counts[mask] / total
    weights[mask] = torch.log(1.0 / freqs + 1)
    mean_w = weights[mask].mean()
    if mean_w > 0:
        weights[mask] /= mean_w
    return weights


def _run_eval(model, loader, dataset_len, device):
    """Run one evaluation pass; return (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            out = model.forward(input_ids=input_ids, labels=labels)
            total_loss += out['loss'].item() * input_ids.size(0)
            preds = torch.argmax(out['logits'][:, -1, :], dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / dataset_len, correct / total


def train_and_evaluate(
    model,
    train_loader,
    eval_loader,
    eval_dataset_len,
    epochs,
    learning_rate,
    weight_decay,
    early_stopping_patience,
    device,
):
    """Train *model* with early stopping; return ``(final_eval_loss, final_eval_accuracy)``.

    The model's weights are updated in-place and the best checkpoint is
    restored before the final evaluation pass.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_eval_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for input_ids, labels in train_loader:
            optimizer.zero_grad()
            input_ids, labels = input_ids.to(device), labels.to(device)
            out = model.forward(input_ids=input_ids, labels=labels)
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += out['loss'].item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_eval_loss, eval_accuracy = _run_eval(model, eval_loader, eval_dataset_len, device)
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f | Eval Loss: %.4f | Eval Accuracy: %.4f",
            epoch + 1, epochs, avg_train_loss, avg_eval_loss, eval_accuracy,
        )

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            epochs_without_improvement = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.get_model().state_dict().items()}
            logger.info("  ✓ New best eval loss: %.4f", best_eval_loss)
        else:
            epochs_without_improvement += 1
            logger.info("  No improvement for %d epoch(s)", epochs_without_improvement)

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            logger.info("Early stopping after epoch %d", epoch + 1)
            break
        model.train()

    if best_model_state is not None:
        model.get_model().load_state_dict(best_model_state)
        logger.info("Restored best model (eval loss: %.4f)", best_eval_loss)

    return _run_eval(model, eval_loader, eval_dataset_len, device)


PreparedData = namedtuple(
    'PreparedData',
    ['processor', 'words', 'vocab', 'train_loader', 'eval_loader', 'train_subset', 'eval_subset'],
)


def prepare_data(
    quotes_dict,
    stock_ids,
    interval_minutes,
    context_window_size,
    batch_size,
    delta_values=None,
    min_words=0,
    min_sequences=0,
):
    """Convert a quotes dict into train/eval DataLoaders.

    Args:
        quotes_dict: dict mapping stock_id -> list of quote dicts.
        stock_ids: ordered list of stock IDs to include.
        interval_minutes: price-sampling interval for PriceProcessor.
        context_window_size: sequence length for StockWordDataset.
        batch_size: DataLoader batch size.
        delta_values: optional custom delta thresholds for PriceProcessor.
        min_words: return None when fewer words are generated.
        min_sequences: return None when the dataset has fewer sequences.

    Returns:
        A PreparedData named tuple, or None if the data does not meet the
        minimum size requirements.
    """
    processor = PriceProcessor(interval_minutes=interval_minutes, delta_values=delta_values)
    words = processor.extract_words(quotes_dict, stock_ids)

    if len(words) < min_words:
        logger.warning(
            "Only %d words generated (need >= %d) — skipping.", len(words), min_words
        )
        return None
    if not words:
        logger.warning("No words generated — skipping.")
        return None

    _, unique_words = processor.count_unique_words(words)
    vocab = {word: idx for idx, word in enumerate(sorted(unique_words))}
    dataset = StockWordDataset(words=words, vocab=vocab, context_window_size=context_window_size)

    if len(dataset) < min_sequences:
        logger.warning(
            "Dataset too small (%d sequences, need >= %d) — skipping.", len(dataset), min_sequences
        )
        return None
    if not dataset:
        logger.warning("Dataset is empty — skipping.")
        return None

    split = int(len(dataset) * 0.85)
    train_subset = torch.utils.data.Subset(dataset, range(0, split))
    eval_subset = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    eval_loader = DataLoader(
        eval_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    return PreparedData(
        processor=processor,
        words=words,
        vocab=vocab,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_subset=train_subset,
        eval_subset=eval_subset,
    )


def _log_delta_distribution(processor, words):
    """Log the per-symbol delta frequency distribution."""
    delta_labels = []
    for i, delta in enumerate(processor.delta_values):
        char = chr(ord('a') + i)
        label = f"{char} (0%)" if delta == 0 else f"{char} ({'+' if delta > 0 else ''}{delta*100:.1f}%)"
        delta_labels.append(label)

    counts = {lbl: 0 for lbl in delta_labels}
    for word in words:
        for ch in word:
            if ch.isalpha():
                idx = ord(ch) - ord('a')
                if 0 <= idx < len(delta_labels):
                    counts[delta_labels[idx]] += 1

    total = sum(counts.values())
    logger.info(f"Total deltas: {total}")
    for lbl, cnt in counts.items():
        pct = cnt / total * 100 if total else 0
        bar = '█' * int(pct / 2)
        logger.info(f"  {lbl:15} {cnt:8} ({pct:6.2f}%) {bar}")


def _log_confusion_matrices(eval_loader, model, vocab, processor, tickers, device):
    """Collect predictions and log per-stock confusion matrices and accuracy summary."""
    idx_to_word = {idx: word for word, idx in vocab.items()}
    delta_letters = [chr(ord('a') + i) for i in range(len(processor.delta_values))]

    all_true, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for input_ids, labels in eval_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            out = model.forward(input_ids=input_ids, labels=labels)
            preds = torch.argmax(out['logits'][:, -1, :], dim=-1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    true_words = [idx_to_word.get(i, '?' * len(tickers)) for i in all_true]
    pred_words = [idx_to_word.get(i, '?' * len(tickers)) for i in all_pred]

    logger.info("")
    logger.info("=" * 60)
    logger.info("PER-STOCK CONFUSION MATRICES (Letter-by-Letter Analysis)")
    logger.info("=" * 60)

    stock_accuracies = []
    for pos, ticker in enumerate(tickers):
        actual = [tw[pos] for tw, pw in zip(true_words, pred_words) if pos < len(tw) and pos < len(pw)]
        predicted = [pw[pos] for tw, pw in zip(true_words, pred_words) if pos < len(tw) and pos < len(pw)]
        if not actual:
            continue

        logger.info(f"  Actual letter distribution: {dict(sorted(Counter(actual).items()))}")

        confusion = {a: {p: 0 for p in delta_letters} for a in delta_letters}
        for a, p in zip(actual, predicted):
            if a in confusion and p in delta_letters:
                confusion[a][p] += 1

        correct = sum(1 for a, p in zip(actual, predicted) if a == p)
        acc = correct / len(actual)
        stock_accuracies.append((ticker, acc))

        logger.info(f"\n{ticker} (position {pos + 1}/{len(tickers)}) - Accuracy: {acc:.4f}")
        header = "Actual\\Pred | " + " | ".join(f" {l} " for l in delta_letters) + " | Total"
        sep = "-" * len(header)
        logger.info(sep)
        logger.info(header)
        logger.info(sep)
        for a in delta_letters:
            row_total = sum(confusion[a].values())
            if row_total:
                row = " | ".join(f"{confusion[a][p]:3d}" for p in delta_letters)
                logger.info(f"     {a}      | {row} | {row_total:5d}")
        logger.info(sep)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Per-Stock Accuracies")
    logger.info("=" * 60)
    for ticker, acc in sorted(stock_accuracies, key=lambda x: x[1], reverse=True):
        logger.info(f"  {ticker:6s}: {acc:.4f}")
    if stock_accuracies:
        avg = sum(a for _, a in stock_accuracies) / len(stock_accuracies)
        logger.info(f"\n  Average per-stock accuracy: {avg:.4f}")
    logger.info("=" * 60)


def main():
    default_config = load_default_config()
    user_config = {}

    args = _build_arg_parser().parse_args()
    if args.config:
        user_config = load_config(args.config)
    _merge_config(args, default_config, user_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info("=" * 60)
    logger.info("RUN PARAMETERS")
    logger.info("=" * 60)
    logger.info(f"Number of stocks:          {args.num_stocks}")
    logger.info(f"Context window size:       {args.context_window_size}")
    logger.info(f"Interval (minutes):        {args.interval}")
    logger.info(f"Epochs:                    {args.epochs}")
    logger.info(f"Batch size:                {args.batch_size}")
    logger.info(f"Learning rate:             {args.learning_rate}")
    if args.start_date or args.end_date:
        logger.info(f"Quote date range:          {args.start_date or 'earliest'} to {args.end_date or 'latest'}")
    logger.info("=" * 60)

    # --- Database & stocks ---
    logger.info("Connecting to database...")
    db = StockDatabase(password=args.db_password)
    db.connect()

    stock_ids, random_stocks = _fetch_stocks(db, args)
    random_stocks = _refresh_stock_names(db, stock_ids)
    stock_ids = [s['id'] for s in random_stocks]

    # --- Quotes ---
    logger.info("Fetching quote data...")
    if args.start_date or args.end_date:
        logger.info(f"  Date range: {args.start_date or 'earliest'} to {args.end_date or 'latest'}")
    data_start = time.time()
    quotes_dict = db.get_quotes_for_stocks(stock_ids, start_date=args.start_date, end_date=args.end_date)
    for sid, quotes in quotes_dict.items():
        ticker = next((s['ticker'] for s in random_stocks if s['id'] == sid), 'Unknown')
        logger.info(f"  {ticker}: {len(quotes)} quotes")

    logger.info("Filtering stocks by data recency (max 30 days old)...")
    stock_ids, random_stocks, quotes_dict = filter_stocks_by_recency(
        db, stock_ids, random_stocks, quotes_dict, max_age_days=30
    )
    logger.info(f"Using {len(stock_ids)} stocks after filtering:")
    for s in random_stocks:
        count = len(quotes_dict.get(s['id'], []))
        logger.info(f"  {s['ticker']}: {count} quotes - {s['name'] or 'N/A'}")

    # --- Data processing ---
    logger.info(f"Processing data with {args.interval}-minute intervals...")
    delta_values = args.delta_ranges
    if delta_values is not None:
        logger.info(f"Using delta ranges: {delta_values}")
    prepared = prepare_data(
        quotes_dict=quotes_dict,
        stock_ids=stock_ids,
        interval_minutes=args.interval,
        context_window_size=args.context_window_size,
        batch_size=args.batch_size,
        delta_values=delta_values,
    )
    if prepared is None:
        logger.error("Data preparation failed — no words generated or dataset too small.")
        return
    processor, words, vocab = prepared.processor, prepared.words, prepared.vocab
    train_dataset, eval_dataset = prepared.train_subset, prepared.eval_subset
    train_loader, eval_loader = prepared.train_loader, prepared.eval_loader

    unique_count = len(vocab)
    logger.info(f"Generated {len(words)} words")
    logger.info(f"Unique words: {unique_count}  ({unique_count / len(words) * 100:.2f}% vocabulary coverage)")
    logger.info("Top 10 most common price movement patterns:")
    for word, count in Counter(words).most_common(10):
        logger.info(f"  '{word}': {count:6} times ({count / len(words) * 100:5.2f}%)")
    logger.info(f"Delta frequency distributions ({args.interval}-minute intervals):")
    _log_delta_distribution(processor, words)

    logger.info(f"Created dataset with {len(train_dataset) + len(eval_dataset)} sequences")
    logger.info(f"Train sequences: {len(train_dataset)}, Eval sequences: {len(eval_dataset)}")

    data_elapsed = time.time() - data_start
    logger.info(f"Data preparation: {data_elapsed:.1f}s ({data_elapsed/60:.2f} min)")

    model = StockTransformerModel(
        vocab_size=len(vocab),
        max_position_embeddings=args.max_position_embeddings,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        device=device,
    )

    if args.use_class_weights:
        logger.info("Computing class weights from training data...")
        weights = _compute_class_weights(train_dataset, len(vocab), device)
        mask = weights > 0
        logger.info(f"Weight range: min={weights[mask].min():.4f}, max={weights[mask].max():.4f}")
        model.set_class_weights(weights)
    else:
        logger.info("Class weighting disabled")

    # --- Training ---
    logger.info(f"Starting training for {args.epochs} epochs...")
    if args.early_stopping_patience > 0:
        logger.info(f"Early stopping patience: {args.early_stopping_patience}")

    model_start = time.time()
    final_eval_loss, final_eval_accuracy = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        eval_dataset_len=len(eval_dataset),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
    )
    training_elapsed = time.time() - model_start
    logger.info(f"Training: {training_elapsed:.1f}s ({training_elapsed/60:.2f} min)")

    final_perplexity = np.exp(final_eval_loss)
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON HELD-OUT EVAL SET")
    logger.info("=" * 60)
    logger.info(f"Final Eval Loss:     {final_eval_loss:.4f}")
    logger.info(f"Final Eval Accuracy: {final_eval_accuracy:.4f}")
    logger.info(f"Final Perplexity:    {final_perplexity:.4f}")
    logger.info("=" * 60)

    tickers = [s['ticker'] for s in random_stocks]
    _log_confusion_matrices(eval_loader, model, vocab, processor, tickers, device)

    logger.info("Analysis complete!")

    # --- Save artifacts ---
    output_dir = log_dir / "output"
    output_dir.mkdir(exist_ok=True)
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)

    config = user_config if user_config else default_config

    if args.save_model:
        save_model(model, models_dir, logger, tickers=tickers, config=config, args=args,
                   log_file=log_file, delta_values=delta_values)
    if args.save_vocabulary:
        save_vocabulary(vocab, output_dir, logger)
    if args.save_predictions:
        _, preds_true, preds_pred = [], [], []
        model.eval()
        with torch.no_grad():
            for input_ids, labels in eval_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                out = model.forward(input_ids=input_ids, labels=labels)
                p = torch.argmax(out['logits'][:, -1, :], dim=-1)
                preds_true.extend(labels.cpu().numpy())
                preds_pred.extend(p.cpu().numpy())
        save_predictions({'true_labels': preds_true, 'pred_labels': preds_pred,
                          'accuracy': final_eval_accuracy}, output_dir, logger)
    if args.plot_results:
        plot_results({'epochs': [], 'losses': []}, {'epochs': [], 'losses': [], 'accuracies': []},
                     output_dir, logger)

    db.close()


if __name__ == '__main__':
    import numpy as np
    main()
