"""Utilities for analyzing and visualizing training results.

Exported classes:
    ModelViewer -- Streamlit-based web UI for browsing trained models under
                   models/: a top-N leaderboard, ticker search, and
                   per-model drilldown.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import json

from pytink.model_registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_PARQUET_PATH,
    records_to_dataframe,
    scan_models_dir,
    search_models_df,
    top_models_df,
)

logger = logging.getLogger(__name__)


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

    Returns:
        Dictionary with keys 'accuracy', 'high_conf_predictions', and
        'high_conf_accuracy'.
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
    """Save vocabulary to JSON file.

    Args:
        vocab: Dictionary mapping word strings to token IDs.
        filepath: Destination file path.
    """
    with open(filepath, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary saved to {filepath}")


def load_vocabulary(filepath: str) -> Dict[str, int]:
    """Load vocabulary from JSON file.

    Args:
        filepath: Path to a JSON file previously written by save_vocabulary.

    Returns:
        Dictionary mapping word strings to token IDs.
    """
    with open(filepath, 'r') as f:
        vocab = json.load(f)
    return vocab


# Columns shown in the leaderboard / search tables, in display order.
_TABLE_COLUMNS = [
    "tickers_display",
    "eval_accuracy",
    "eval_loss",
    "perplexity",
    "metrics_source",
    "interval_minutes",
    "context_window_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "max_position_embeddings",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "early_stopping_patience",
    "run_timestamp",
]


def _directory_signature(models_dir: Path):
    """A cheap (count, max-mtime) fingerprint used to invalidate the cache
    when models are added/removed, without rescanning the full tree."""
    if not models_dir.exists():
        return (0, 0.0)
    paths = list(models_dir.glob("*/*/config.yaml"))
    if not paths:
        return (0, 0.0)
    return (len(paths), max(p.stat().st_mtime for p in paths))


class ModelViewer:
    """Streamlit web UI for browsing trained pytink models.

    Reads model configuration, training logs, and the farm's aggregated
    ``models.parquet`` from the ``models/`` directory tree (see
    :mod:`pytink.model_registry`) and presents:

    - A top-N leaderboard ranked by evaluation accuracy, showing every
      recorded metric, ticker list, and training/config parameter.
    - A ticker search that lists all models trained on a given set of
      tickers (order-independent, all must be present).
    - A per-model drilldown with the full config, per-stock accuracy
      (when available), and other training details.

    This class only depends on ``streamlit`` at render time, so importing
    :mod:`pytink.analysis` does not require streamlit to be installed.

    Usage:
        # streamlit run src/pytink/viewer_app.py
        from pytink.analysis import ModelViewer
        ModelViewer().render()
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        parquet_path: Optional[Path] = None,
        default_top_n: int = 10,
    ):
        """Initialise a ModelViewer.

        Args:
            models_dir: Root models directory (default: ``<repo>/models``).
            parquet_path: Path to the farm's aggregated metrics log
                (default: ``<repo>/models.parquet``).
            default_top_n: Default number of models shown in the leaderboard
                (default: 10).
        """
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.parquet_path = Path(parquet_path) if parquet_path else DEFAULT_PARQUET_PATH
        self.default_top_n = default_top_n

    def _load_dataframe(self):
        """Scan models_dir (cached by Streamlit until its contents change)."""
        import streamlit as st

        @st.cache_data(show_spinner="Scanning models directory...")
        def _cached(models_dir_str, parquet_path_str, _signature):
            records = scan_models_dir(Path(models_dir_str), Path(parquet_path_str))
            return records_to_dataframe(records)

        signature = _directory_signature(self.models_dir)
        return _cached(str(self.models_dir), str(self.parquet_path), signature)

    def render(self):
        """Render the full viewer page. Call once per Streamlit script run."""
        import streamlit as st

        st.set_page_config(page_title="pytink Model Viewer", layout="wide")
        st.title("pytink Model Viewer")
        st.caption(str(self.models_dir))

        if st.sidebar.button("Refresh data"):
            st.cache_data.clear()

        df = self._load_dataframe()
        if df.empty:
            st.warning(f"No trained models found under {self.models_dir}")
            return

        st.sidebar.header("Top Models")
        top_n = st.sidebar.number_input(
            "Number of top models", min_value=1, max_value=len(df),
            value=min(self.default_top_n, len(df)),
        )

        st.sidebar.header("Search by Tickers")
        ticker_input = st.sidebar.text_input(
            "Tickers (comma-separated)", "", help="e.g. AAPL, MSFT"
        )

        tab_top, tab_search = st.tabs(["Top Models", "Search by Tickers"])

        with tab_top:
            st.subheader(f"Top {top_n} Models by Eval Accuracy")
            self._render_table_with_drilldown(top_models_df(df, n=top_n), key_prefix="top")

        with tab_search:
            tickers = [t for t in (s.strip() for s in ticker_input.split(",")) if t]
            if tickers:
                st.subheader(f"Models containing: {', '.join(tickers)}")
                result_df = search_models_df(df, tickers)
                st.write(f"{len(result_df)} matching model(s)")
                self._render_table_with_drilldown(result_df, key_prefix="search")
            else:
                st.info("Enter one or more tickers in the sidebar to search.")

    def _render_table_with_drilldown(self, df, key_prefix: str):
        """Render a summary table plus a selectbox-driven detail view."""
        import streamlit as st

        if df.empty:
            st.write("No matching models.")
            return

        st.dataframe(df[_TABLE_COLUMNS], width="stretch", hide_index=True)

        options = df["model_dir"].tolist()
        labels = {
            row["model_dir"]: f"{row['tickers_display']}  ({row['run_timestamp']})"
            for _, row in df.iterrows()
        }
        selected = st.selectbox(
            "Select a model for details",
            options,
            format_func=lambda d: labels.get(d, d),
            key=f"{key_prefix}_select",
        )
        if selected:
            self._render_detail(df[df["model_dir"] == selected].iloc[0])

    def _render_detail(self, row):
        """Render the drilldown panel for a single model record."""
        import streamlit as st
        import pandas as pd

        st.markdown(f"#### {row['tickers_display']}")
        st.caption(f"{row['model_dir']}  —  source: {row['metrics_source']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Eval Accuracy", f"{row['eval_accuracy']:.4f}" if pd.notna(row["eval_accuracy"]) else "N/A")
        col2.metric("Eval Loss", f"{row['eval_loss']:.4f}" if pd.notna(row["eval_loss"]) else "N/A")
        col3.metric("Perplexity", f"{row['perplexity']:.4f}" if pd.notna(row["perplexity"]) else "N/A")

        if row["per_stock_accuracy"]:
            st.markdown("**Per-stock accuracy**")
            per_stock_df = pd.DataFrame(
                sorted(row["per_stock_accuracy"].items(), key=lambda kv: -kv[1]),
                columns=["ticker", "accuracy"],
            )
            st.dataframe(per_stock_df, width="stretch", hide_index=True)

        st.markdown("**Full configuration**")
        st.json(row["raw_config"])

    def run(self):
        """Alias for :meth:`render`, for symmetry with other pytink entry points."""
        self.render()
