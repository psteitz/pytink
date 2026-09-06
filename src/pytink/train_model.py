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
        """Determine whether the given log record should be emitted.

        Suppresses per-batch progress messages (those whose text contains the
        word ``"Batch"``), allowing only epoch-level summaries to pass through.

        Args:
            record: The :class:`logging.LogRecord` instance to evaluate.

        Returns:
            False if the record's message contains ``"Batch"``; True otherwise.
        """
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


def filter_stocks_by_recency(db, stock_ids, random_stocks, quotes_dict, max_age_days=30, _depth=0, _max_depth=5):
    """Filter out stocks with stale data (last quote older than max_age_days).

    Stocks whose most recent quote pre-dates ``max_age_days`` before the
    most recent quote timestamp are removed. Removed stocks are replaced by
    fetching fresh candidates from the database; the replacement candidates
    are themselves recursively passed through this same function to ensure
    they also meet the recency requirement.

    Args:
        db: StockDatabase instance used to fetch
            replacement stocks when stale ones are removed.
        stock_ids: Ordered list of integer stock IDs to evaluate.
        random_stocks: List of stock dicts, each containing at least the keys
            ``'id'``, ``'ticker'``, and ``'name'``.
        quotes_dict: Dict mapping each stock ID to a list of quote dicts.
            Each quote dict must contain a ``'timestamp'`` key whose value is
            either a datetime or an ISO-format date string.
        max_age_days: A stock is considered stale when its most recent quote
            is more than this many days older than the newest quote found
            across all stocks in *quotes_dict* (``max_timestamp`` in the
            function body).  Defaults to 30.

    Returns:
        A 3-tuple ``(filtered_stock_ids, filtered_random_stocks,
        filtered_quotes_dict)`` with the same structure as the corresponding
        inputs but containing only stocks that passed the recency check (plus
        any fresh replacements).
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
    
    # Create lookup dicts
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
            if _depth >= _max_depth:
                logger.warning(
                    f"filter_stocks_by_recency: reached max recursion depth ({_max_depth}); "
                    "accepting remaining replacements without further recency check"
                )
                replacement_ids = [s['id'] for s in unique_replacements]
                replacement_quotes = db.get_quotes_for_stocks(replacement_ids)
                replacement_ids, replacement_objs, replacement_q = (
                    replacement_ids, unique_replacements, replacement_quotes
                )
            else:
                replacement_ids = [s['id'] for s in unique_replacements]
                replacement_quotes = db.get_quotes_for_stocks(replacement_ids)

                # Recursively filter replacements
                replacement_ids, replacement_objs, replacement_q = filter_stocks_by_recency(
                    db, replacement_ids, unique_replacements, replacement_quotes,
                    max_age_days, _depth + 1, _max_depth
                )
            
            valid_stock_ids.extend(replacement_ids)
            valid_random_stocks.extend(replacement_objs)
            valid_quotes_dict.update(replacement_q)
    
    return valid_stock_ids, valid_random_stocks, valid_quotes_dict


def load_config(config_path):
    """Load configuration from a YAML file.

    Args:
        config_path: Path-like object or string pointing to a YAML config file.

    Returns:
        A dictionary containing the parsed configuration values, or an empty
        dict if PyYAML is not installed or the file cannot be read.
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
    """Load default configuration from ``config_template.yaml``.

    Looks for the template file in the same directory as this script.
    Falls back to an empty dict if the file does not exist.

    Returns:
        A dictionary containing the default configuration values parsed from
        ``config_template.yaml``, or an empty dict if the file is absent.
    """
    template_path = Path(__file__).parent / 'config_template.yaml'
    if template_path.exists():
        return load_config(template_path)
    else:
        logger.warning(f"Default config template not found at {template_path}")
        return {}


def save_model(model, output_dir, logger, tickers=None, config=None, args=None, log_file=None, delta_values=None,
               training_start_date=None, training_end_date=None):
    """Save the trained model weights, a reproducible config, and the training log.

    Artifacts are written under ``output_dir/<tickers>/<timestamp>/``, e.g.
    ``models/AAPL-GOOGL-MSFT/20260101_143052/``.  The directory is created
    if it does not already exist.

    Args:
        model: Trained :class:`~pytink.model.StockTransformerModel` instance
            whose weights will be serialised to ``model.pt``.
        output_dir: Base directory under which the ticker/timestamp
            subdirectory structure is created.
        logger: :class:`logging.Logger` instance used for status and error
            messages.
        tickers: Optional list of ticker symbol strings.  When provided they
            are sorted and joined with ``'-'`` to form the subdirectory name.
            Defaults to ``"model"`` when ``None``.
        config: Optional configuration dict used during training.  Used only
            as a fallback source for ``delta_ranges`` when *delta_values* is
            ``None``.
        args: Optional :class:`argparse.Namespace` carrying resolved training
            hyper-parameters (e.g. ``batch_size``, ``epochs``, ``learning_rate``).
            When ``None``, hard-coded defaults are written to the config file.
        log_file: Optional path to the training log file to copy into the
            output directory as ``training.log``.
        delta_values: Optional list of numeric delta threshold values used by
            :class:`~pytink.processor.PriceProcessor` to encode price movements.
            Written to the saved config under ``delta_ranges``.
        training_start_date: Optional :class:`~datetime.datetime` (or ISO-format
            string) representing the earliest quote timestamp present in the
            training data.  Saved to the config as ``data.start_date``.
        training_end_date: Optional :class:`~datetime.datetime` (or ISO-format
            string) representing the latest quote timestamp present in the
            training data.  Saved to the config as ``data.end_date``.
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
        
        # Add actual training date range if available
        if training_start_date is not None:
            training_config['data']['start_date'] = (
                training_start_date.strftime('%Y-%m-%d')
                if isinstance(training_start_date, datetime)
                else str(training_start_date)
            )
        if training_end_date is not None:
            training_config['data']['end_date'] = (
                training_end_date.strftime('%Y-%m-%d')
                if isinstance(training_end_date, datetime)
                else str(training_end_date)
            )

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
    """Save the vocabulary mapping to a JSON file.

    Writes ``vocabulary.json`` into *output_dir*.  The file contains a
    JSON object mapping each word string to its integer token ID.

    Args:
        vocab: Dictionary mapping word strings to integer token IDs.
        output_dir: Directory in which to write ``vocabulary.json``.
            The directory must already exist.
        logger: :class:`logging.Logger` instance for status and error messages.
    """
    output_path = Path(output_dir) / "vocabulary.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        logger.info(f"✓ Vocabulary saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save vocabulary: {e}")


def save_predictions(predictions, output_dir, logger):
    """Save model predictions to a JSON file.

    Writes ``predictions.json`` into *output_dir*, converting any NumPy
    arrays or tensors to plain Python lists for JSON compatibility.

    Args:
        predictions: Dict containing prediction data.  Recognised keys:

            * ``'true_labels'`` – iterable of ground-truth integer class IDs.
            * ``'pred_labels'`` – iterable of predicted integer class IDs.
            * ``'sequences'``   – optional list of raw input sequences.
            * ``'accuracy'``    – optional scalar overall accuracy value.

        output_dir: Directory in which to write ``predictions.json``.
            The directory must already exist.
        logger: :class:`logging.Logger` instance for status and error messages.
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
    """Generate and save training/evaluation result plots.

    Produces a two-panel figure:  the left panel shows training and
    evaluation loss curves; the right panel shows evaluation accuracy.
    The figure is saved as ``results.png`` inside *output_dir*.

    Args:
        training_history: Dict with the following keys:

            * ``'epochs'`` – list of epoch numbers for which training loss
              was recorded.
            * ``'losses'`` – corresponding list of average training loss
              values.

        eval_history: Dict with the following keys:

            * ``'epochs'``     – list of epoch numbers for evaluation.
            * ``'losses'``     – corresponding list of evaluation loss values.
            * ``'accuracies'`` – corresponding list of evaluation accuracy
              values.

        output_dir: Path-like object for the directory in which
            ``results.png`` will be written.  Created if absent.
        logger: :class:`logging.Logger` instance for status and error messages.
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
    """Parse a ``YYYY-MM-DD`` string into a :class:`~datetime.datetime`.

    Args:
        val: Date string in ``YYYY-MM-DD`` format, or ``None``.
        name: Human-readable parameter name used in the error message when
            *val* cannot be parsed (e.g. ``'--start-date'``).

    Returns:
        A :class:`~datetime.datetime` corresponding to *val*, or ``None`` if
        *val* is ``None``.

    Raises:
        SystemExit: Calls :func:`sys.exit` with exit code 1 when *val* is not
            ``None`` but does not match the ``YYYY-MM-DD`` format.
    """
    if val is None:
        return None
    try:
        return datetime.strptime(val, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid {name} format '{val}' — expected YYYY-MM-DD")
        sys.exit(1)


def _build_arg_parser():
    """Build and return the command-line argument parser.

    Defines all CLI flags accepted by :func:`main`, including database
    credentials, data selection, model architecture, and training
    hyper-parameters.  Default values are intentionally left as ``None``
    so that :func:`_merge_config` can apply the correct priority order.

    Returns:
        A configured :class:`argparse.ArgumentParser` instance ready to be
        called with ``.parse_args()``.
    """
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
    """Resolve all configuration values and write them back onto *args* in-place.

    Priority order (highest to lowest):
    CLI argument > user config file > default config template > hard-coded fallback.

    Args:
        args: :class:`argparse.Namespace` produced by :func:`_build_arg_parser`.
            Attributes are mutated in-place with the resolved values.
        default_config: Dict loaded from ``config_template.yaml`` via
            :func:`load_default_config`; used as the penultimate fallback.
        user_config: Dict loaded from the user-supplied ``--config`` file via
            :func:`load_config`; takes precedence over *default_config* when
            a CLI argument is absent.
    """
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
    """Fetch the target stock list from the database, deduplicated by ID.

    When ``args.tickers`` is set, the listed tickers are looked up by symbol
    and the script exits if any are missing.  Otherwise, ``args.num_stocks``
    random stocks with at least 100 000 quotes are returned.

    Args:
        db: Connected :class:`~pytink.database.StockDatabase` instance.
        args: Resolved :class:`argparse.Namespace` with at least the
            attributes ``tickers`` (list or ``None``) and ``num_stocks`` (int).

    Returns:
        A 2-tuple ``(stock_ids, unique_stocks)`` where *stock_ids* is a list
        of integer database IDs and *unique_stocks* is a list of stock dicts
        (each with keys ``'id'``, ``'ticker'``, ``'name'``), both in the same
        order and free of duplicates.
    """
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
    """Fill in any missing stock names via yFinance and return updated rows.

    Calls :meth:`~pytink.database.StockDatabase.update_missing_stock_names`
    to populate the database, then re-queries and logs each stock's ticker
    and name.

    Args:
        db: Connected :class:`~pytink.database.StockDatabase` instance.
        stock_ids: List of integer stock IDs whose names should be refreshed.

    Returns:
        List of stock dicts (``'id'``, ``'ticker'``, ``'name'``) re-fetched
        from the database after the name update, in an unspecified order.
    """
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
    """Compute log-scaled, normalised per-class loss weights from training labels.

    For each class present in *train_dataset* a weight proportional to
    ``log(total / class_count + 1)`` is computed, then the weights are
    normalised so their mean equals 1.  Classes with zero occurrences
    receive a weight of 0.

    Args:
        train_dataset: :class:`torch.utils.data.Subset` (or any dataset)
            whose items are ``(input_ids, label)`` pairs.  The *label*
            must be a scalar integer tensor.
        vocab_size: Total number of classes (vocabulary size); determines
            the length of the returned weight tensor.
        device: Torch device string or object (e.g. ``'cpu'``, ``'cuda'``)
            on which the weight tensor should be placed.

    Returns:
        A 1-D :class:`torch.Tensor` of shape ``(vocab_size,)`` containing
        the normalised class weights, suitable for passing to
        :class:`torch.nn.CrossEntropyLoss` as the ``weight`` argument.
    """
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
    """Run one full evaluation pass over *loader* without gradient updates.

    Sets the model to evaluation mode, iterates over every batch in *loader*,
    accumulates loss and correct-prediction counts, then returns aggregate
    metrics.  The model is left in evaluation mode after the call.

    Args:
        model: :class:`~pytink.model.StockTransformerModel` instance to
            evaluate.  Its ``forward`` method must accept keyword arguments
            ``input_ids`` and ``labels`` and return a dict with keys
            ``'loss'`` and ``'logits'``.
        loader: :class:`torch.utils.data.DataLoader` yielding
            ``(input_ids, labels)`` batches from the evaluation split.
        dataset_len: Total number of samples in the evaluation split, used
            as the denominator when averaging the accumulated loss.
        device: Torch device string or object to which batches are moved
            before being passed to the model.

    Returns:
        An ordered pair ``(avg_loss, accuracy)`` where *avg_loss* is the
        per-sample mean cross-entropy loss (float) and *accuracy* is the
        fraction of correctly predicted next tokens (float in ``[0, 1]``).
    """
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
    """Train *model* with early stopping and return final evaluation metrics.

    Uses the Adam optimiser with gradient clipping (max norm 1.0).  After
    each epoch the model is evaluated on *eval_loader*; the best checkpoint
    (lowest eval loss) is saved in memory and restored before the final
    evaluation pass.  The model's weights are updated in-place.

    Args:
        model: :class:`~pytink.model.StockTransformerModel` instance to train.
            Its ``forward`` method must accept ``input_ids`` and ``labels``
            keyword arguments and return a dict with keys ``'loss'`` and
            ``'logits'``.
        train_loader: :class:`torch.utils.data.DataLoader` for the training
            split, yielding ``(input_ids, labels)`` batches.
        eval_loader: :class:`torch.utils.data.DataLoader` for the evaluation
            split, yielding ``(input_ids, labels)`` batches.
        eval_dataset_len: Total number of samples in the evaluation split;
            used to compute the per-sample mean eval loss.
        epochs: Maximum number of training epochs to run.
        learning_rate: Initial learning rate passed to :class:`torch.optim.Adam`.
        weight_decay: L2 regularisation coefficient passed to
            :class:`torch.optim.Adam`.
        early_stopping_patience: Number of consecutive epochs without an
            improvement in eval loss before training is stopped early.
            Pass ``0`` to disable early stopping.
        device: Torch device string or object to which batches are moved
            before being passed to the model.

    Returns:
        A 2-tuple ``(final_eval_loss, final_eval_accuracy)`` evaluated on
        *eval_loader* using the best model checkpoint found during training.
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
    """Convert a quotes dict into train/eval DataLoaders ready for model training.

    Runs the full preprocessing pipeline: price-word extraction via
    :class:`~pytink.processor.PriceProcessor`, vocabulary construction,
    :class:`~pytink.model.StockWordDataset` creation, an 85/15 train/eval
    split, and :class:`~torch.utils.data.DataLoader` wrapping.

    Args:
        quotes_dict: Dict mapping each stock ID (int) to a list of quote
            dicts.  Each quote dict must contain at least ``'timestamp'``
            and ``'close'`` keys.
        stock_ids: Ordered list of integer stock IDs to include.  The order
            determines the character position of each stock in the generated
            price-movement words.
        interval_minutes: Sampling interval in minutes passed to
            :class:`~pytink.processor.PriceProcessor` for resampling raw
            tick data.
        context_window_size: Number of tokens (price-movement words) in each
            input sequence fed to the model.
        batch_size: Number of sequences per mini-batch for both DataLoaders.
        delta_values: Optional list of numeric delta thresholds passed to
            :class:`~pytink.processor.PriceProcessor`.  When ``None`` the
            processor uses its built-in defaults.
        min_words: Minimum number of words that must be generated before
            proceeding.  Returns ``None`` when the count is below this
            threshold.  Defaults to ``0`` (no minimum).
        min_sequences: Minimum number of sequences the
            :class:`~pytink.model.StockWordDataset` must contain before
            proceeding.  Returns ``None`` when the count is below this
            threshold.  Defaults to ``0`` (no minimum).

    Returns:
        A :class:`PreparedData` named tuple with fields
        ``(processor, words, vocab, train_loader, eval_loader,
        train_subset, eval_subset)``, or ``None`` if the generated data
        does not satisfy *min_words* or *min_sequences*.
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
    """Log the frequency distribution of each delta symbol across all words.

    For every delta threshold defined in ``processor.delta_values`` a
    corresponding letter (``'a'``, ``'b'``, ...) is counted across all
    characters of every word in *words*.  The resulting counts, percentages,
    and a simple bar chart are written to the logger at INFO level.

    Args:
        processor: :class:`~pytink.processor.PriceProcessor` instance whose
            ``delta_values`` attribute defines the ordered list of thresholds
            (and therefore the mapping from letter to percentage change).
        words: List of price-movement word strings as produced by
            :meth:`~pytink.processor.PriceProcessor.extract_words`.
    """
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
    """Log per-stock confusion matrices and an accuracy summary to the logger.

    Runs a full inference pass over *eval_loader*, maps integer predictions
    back to word strings, then—for each stock position in the word—tabulates
    a confusion matrix over delta letters and reports the per-position
    accuracy.  A ranked summary of per-stock accuracies is logged at the end.

    Args:
        eval_loader: :class:`torch.utils.data.DataLoader` for the held-out
            evaluation split, yielding ``(input_ids, labels)`` batches.
        model: :class:`~pytink.model.StockTransformerModel` instance used for
            inference.  Set to evaluation mode internally.
        vocab: Dict mapping word strings to integer token IDs, as produced by
            :func:`prepare_data`.
        processor: :class:`~pytink.processor.PriceProcessor` instance whose
            ``delta_values`` attribute defines the ordered set of delta
            thresholds (and therefore the set of valid delta letters).
        tickers: Ordered list of ticker symbol strings corresponding to stock
            positions within each word (position 0 → tickers[0], etc.).
        device: Torch device string or object to which batches are moved
            before being passed to the model.
    """
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
    """Entry point: parse arguments, train the model, and save all artifacts.

    Orchestrates the full training pipeline:

    1. Parse CLI arguments and merge with config files.
    2. Connect to the database and fetch/deduplicate stocks.
    3. Download quotes, apply recency filtering, and refresh stock names.
    4. Run :func:`prepare_data` to build train/eval DataLoaders.
    5. Construct :class:`~pytink.model.StockTransformerModel` and optionally
       apply class-frequency-based loss weighting.
    6. Call :func:`train_and_evaluate` with early stopping.
    7. Log final metrics, per-stock confusion matrices, and delta distributions.
    8. Optionally save the model, vocabulary, predictions, and result plots.
    """
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

    # Determine actual quote date range included in training data
    training_start_date = None
    training_end_date = None
    for quotes in quotes_dict.values():
        for quote in quotes:
            ts = quote['timestamp']
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if training_start_date is None or ts < training_start_date:
                training_start_date = ts
            if training_end_date is None or ts > training_end_date:
                training_end_date = ts

    # --- Save artifacts ---
    output_dir = log_dir / "output"
    output_dir.mkdir(exist_ok=True)
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)

    config = user_config if user_config else default_config

    if args.save_model:
        save_model(model, models_dir, logger, tickers=tickers, config=config, args=args,
                   log_file=log_file, delta_values=processor.delta_values,
                   training_start_date=training_start_date, training_end_date=training_end_date)
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
