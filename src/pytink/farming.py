#!/usr/bin/env python3
"""
Model farming: generates, trains, and evaluates random stock prediction models,
keeping a pool of the best performers.

Usage:
    python -m pytink.farming --db-password YOUR_PASSWORD
    python -m pytink.farming --db-password YOUR_PASSWORD --num-models 50 --num-generations 5
"""

import sys
import argparse
import logging
import random
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import torch
import numpy as np

MODELS_PARQUET_PATH = Path(__file__).parent.parent.parent / "models.parquet"

from pytink.database import StockDatabase
from pytink.model import StockTransformerModel
from pytink.train_model import prepare_data, train_and_evaluate

logger = logging.getLogger(__name__)

# ── Configurable defaults ────────────────────────────────────────────────────
NUM_MODELS = 100          # Pool size (number of models to maintain)
NUM_GENERATIONS = 10      # Generational cycles after the cold start
DISPLAY_TOP_N = 10        # Number of top models to display at the end

# Stock selection
MIN_STOCKS = 5            # Minimum stocks per model
MAX_STOCKS = 15           # Maximum stocks per model
MIN_QUOTES = 100_000      # Minimum quote count required for a stock to be eligible

# Data processing
DEFAULT_INTERVAL_MINUTES = 30
DEFAULT_CONTEXT_WINDOW_SIZE = 16

# Training
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5        # Epochs per farm model (fewer than full train for speed)
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EARLY_STOPPING_PATIENCE = 3

# Model architecture (fixed for fair comparison across the pool)
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYERS = 4
NUM_ATTENTION_HEADS = 4
MAX_POSITION_EMBEDDINGS = 256
# ─────────────────────────────────────────────────────────────────────────────


class ModelEntry:
    """A trained model paired with its evaluation metrics and metadata."""

    def __init__(
        self,
        model: StockTransformerModel,
        tickers: List[str],
        stock_ids: List[int],
        vocab: Dict[str, int],
        eval_loss: float,
        eval_accuracy: float,
        num_words: int,
        generation: int = 0,
    ):
        self.model = model
        self.tickers = tickers
        self.stock_ids = stock_ids
        self.vocab = vocab
        self.eval_loss = eval_loss
        self.eval_accuracy = eval_accuracy
        self.perplexity = float(np.exp(min(eval_loss, 88)))  # cap to avoid overflow
        self.num_words = num_words
        self.generation = generation
        self.created_at = datetime.now()

    def __repr__(self) -> str:
        return (
            f"ModelEntry(tickers={self.tickers}, "
            f"accuracy={self.eval_accuracy:.4f}, "
            f"loss={self.eval_loss:.4f}, "
            f"perplexity={self.perplexity:.4f}, "
            f"generation={self.generation})"
        )


class ModelFarm:
    """
    Grows and evolves a pool of random stock prediction models.

    Lifecycle
    ---------
    1. cold_start()  — populate the pool with NUM_MODELS randomly generated models.
    2. run()         — cold start, then run NUM_GENERATIONS replacement cycles,
                       each keeping the top 25 % and replacing the bottom 75 %
                       with newly generated random models.
    3. display_top_models() — print a sorted leaderboard.
    """

    def __init__(
        self,
        db_password: str,
        db_host: str = "localhost",
        db_user: str = "tinker",
        db_name: str = "tinker",
        num_models: int = NUM_MODELS,
        num_generations: int = NUM_GENERATIONS,
        min_stocks: int = MIN_STOCKS,
        max_stocks: int = MAX_STOCKS,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        context_window_size: int = DEFAULT_CONTEXT_WINDOW_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
        device: Optional[str] = None,
    ):
        self.num_models = num_models
        self.num_generations = num_generations
        self.min_stocks = min_stocks
        self.max_stocks = max_stocks
        self.interval_minutes = interval_minutes
        self.context_window_size = context_window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.models: List[ModelEntry] = []
        self.generation: int = 0

        logger.info("Connecting to database...")
        self.db = StockDatabase(
            password=db_password,
            host=db_host,
            user=db_user,
            database=db_name,
        )
        self.db.connect()

        # Available stocks are fetched once and cached
        self._eligible_stocks: Optional[List[Dict]] = None

        logger.info(
            "ModelFarm initialised: num_models=%d, num_generations=%d, device=%s",
            num_models, num_generations, self.device,
        )

    # ── Stock pool helpers ───────────────────────────────────────────────────

    def _get_eligible_stocks(self) -> List[Dict]:
        """Fetch and cache all eligible stocks (sufficient quote depth)."""
        if self._eligible_stocks is None:
            logger.info(
                "Fetching eligible stocks (min %d quotes)...", MIN_QUOTES
            )
            # Request a large sample so we have a diverse population to draw from.
            self._eligible_stocks = self.db.get_random_stocks(
                count=1000, min_quotes=MIN_QUOTES
            )
            logger.info(
                "Cached %d eligible stocks.", len(self._eligible_stocks)
            )
        return self._eligible_stocks

    # ── Single-model pipeline ────────────────────────────────────────────────

    def _build_and_evaluate(self, generation: int) -> Optional[ModelEntry]:
        """
        Select random stocks, train a model, evaluate it, and return a
        ModelEntry.  Returns None when data is insufficient.
        """
        eligible = self._get_eligible_stocks()
        if not eligible:
            logger.warning("No eligible stocks found.")
            return None

        # Random stock count and selection
        num_stocks = random.randint(self.min_stocks, min(self.max_stocks, len(eligible)))
        selected = random.sample(eligible, num_stocks)
        stock_ids = [s["id"] for s in selected]
        tickers = [s["ticker"] for s in selected]

        logger.info(
            "  [gen %d] Training on %d stocks: %s",
            generation, len(tickers), tickers,
        )

        # Fetch quotes
        quotes_dict = self.db.get_quotes_for_stocks(stock_ids)

        # Drop any stocks with no data
        valid_ids = [
            sid for sid in stock_ids
            if sid in quotes_dict and quotes_dict[sid]
        ]
        if len(valid_ids) < 2:
            logger.warning("  Insufficient quote data — skipping.")
            return None

        tickers = [s["ticker"] for s in selected if s["id"] in valid_ids]
        stock_ids = valid_ids
        quotes_dict = {sid: quotes_dict[sid] for sid in stock_ids}

        prepared = prepare_data(
            quotes_dict=quotes_dict,
            stock_ids=stock_ids,
            interval_minutes=self.interval_minutes,
            context_window_size=self.context_window_size,
            batch_size=self.batch_size,
            min_words=self.context_window_size + 10,
            min_sequences=10,
        )
        if prepared is None:
            return None
        words, vocab = prepared.words, prepared.vocab
        train_loader, eval_loader = prepared.train_loader, prepared.eval_loader
        eval_subset = prepared.eval_subset

        # Initialise transformer model
        model = StockTransformerModel(
            vocab_size=len(vocab),
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            device=self.device,
        )

        final_loss, final_accuracy = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            eval_dataset_len=len(eval_subset),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            early_stopping_patience=self.early_stopping_patience,
            device=self.device,
        )

        logger.info(
            "  -> loss=%.4f  accuracy=%.4f  words=%d",
            final_loss, final_accuracy, len(words),
        )

        return ModelEntry(
            model=model,
            tickers=tickers,
            stock_ids=stock_ids,
            vocab=vocab,
            eval_loss=final_loss,
            eval_accuracy=final_accuracy,
            num_words=len(words),
            generation=generation,
        )

    # ── Parquet logging ──────────────────────────────────────────────────────

    def _append_to_parquet(self, entry: ModelEntry) -> None:
        """Append a trained model's metadata and parameters to models.parquet."""
        row = {
            "tickers": "-".join(entry.tickers),
            "accuracy": entry.eval_accuracy,
            "loss": entry.eval_loss,
            "perplexity": entry.perplexity,
            "interval_minutes": self.interval_minutes,
            "context_window_size": self.context_window_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "created_at": entry.created_at,
        }
        df_new = pd.DataFrame([row])
        if MODELS_PARQUET_PATH.exists():
            df_existing = pd.read_parquet(MODELS_PARQUET_PATH)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_parquet(MODELS_PARQUET_PATH, index=False)
        logger.debug("Appended model entry to %s", MODELS_PARQUET_PATH)

    # ── Pool management ──────────────────────────────────────────────────────

    def _sort_pool(self) -> None:
        """Sort the model pool: highest accuracy first, lowest loss as tiebreaker."""
        self.models.sort(key=lambda m: (-m.eval_accuracy, m.eval_loss))

    def cold_start(self) -> None:
        """
        Cold start: generate and evaluate NUM_MODELS random models from scratch,
        populating the model pool.
        """
        logger.info("=" * 60)
        logger.info("COLD START — generating %d random models", self.num_models)
        logger.info("=" * 60)

        t0 = time.time()
        self.models = []

        for i in range(self.num_models):
            logger.info("Cold start model %d/%d:", i + 1, self.num_models)
            entry = self._build_and_evaluate(generation=0)
            if entry is not None:
                self.models.append(entry)
                self._append_to_parquet(entry)

        self._sort_pool()
        elapsed = time.time() - t0
        logger.info(
            "Cold start complete: %d/%d models trained in %.1f s (%.1f min).",
            len(self.models), self.num_models, elapsed, elapsed / 60,
        )
        if self.models:
            logger.info("Best after cold start: %s", self.models[0])

    def _run_generation(self) -> None:
        """
        One generational cycle: keep the top 25 % of the pool and replace the
        bottom 75 % with newly generated random models.
        """
        self.generation += 1

        keep_count = max(1, len(self.models) // 4)
        replace_count = self.num_models - keep_count

        logger.info("=" * 60)
        logger.info(
            "GENERATION %d/%d — keeping top %d, replacing %d",
            self.generation, self.num_generations, keep_count, replace_count,
        )
        logger.info("=" * 60)

        # Retain only the top quarter
        self.models = self.models[:keep_count]

        # Fill the pool back to num_models with new random models
        for i in range(replace_count):
            logger.info(
                "  Generation %d — new model %d/%d:",
                self.generation, i + 1, replace_count,
            )
            entry = self._build_and_evaluate(generation=self.generation)
            if entry is not None:
                self.models.append(entry)
                self._append_to_parquet(entry)

        self._sort_pool()

        if self.models:
            best = self.models[0]
            logger.info(
                "Generation %d complete — best: accuracy=%.4f  tickers=%s",
                self.generation, best.eval_accuracy, best.tickers,
            )

    # ── Output ───────────────────────────────────────────────────────────────

    def display_top_models(self, n: int = DISPLAY_TOP_N) -> None:
        """Print a ranked leaderboard of the top N models."""
        top = self.models[:n]

        print()
        print("=" * 72)
        print(f"TOP {min(n, len(top))} MODELS")
        print("=" * 72)
        print(
            f"{'Rank':<5} {'Accuracy':>9} {'Loss':>9} {'Perplexity':>11}"
            f" {'Gen':>4} {'#':>3}  Tickers"
        )
        print("-" * 72)

        for rank, entry in enumerate(top, start=1):
            tickers_str = "-".join(entry.tickers)
            print(
                f"{rank:<5} {entry.eval_accuracy:>9.4f} {entry.eval_loss:>9.4f}"
                f" {entry.perplexity:>11.4f} {entry.generation:>4}"
                f" {len(entry.tickers):>3}  {tickers_str}"
            )

        print("=" * 72)

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the full farming pipeline: cold start → generations → display."""
        try:
            self.cold_start()

            for _ in range(self.num_generations):
                self._run_generation()

            self.display_top_models()
        finally:
            self.db.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Model Farming: generate and evolve a pool of stock prediction models, "
            "keeping the best performers."
        )
    )
    parser.add_argument(
        "--db-password", type=str, required=True,
        help="Database password (required)",
    )
    parser.add_argument(
        "--db-host", type=str, default="localhost",
        help="Database host (default: localhost; try 127.0.0.1 if localhost auth fails)",
    )
    parser.add_argument(
        "--db-user", type=str, default="tinker",
        help="Database user (default: tinker)",
    )
    parser.add_argument(
        "--db-name", type=str, default="tinker",
        help="Database name (default: tinker)",
    )
    parser.add_argument(
        "--num-models", type=int, default=NUM_MODELS,
        help=f"Pool size — number of models to maintain (default: {NUM_MODELS})",
    )
    parser.add_argument(
        "--num-generations", type=int, default=NUM_GENERATIONS,
        help=f"Number of generational replacement cycles (default: {NUM_GENERATIONS})",
    )
    parser.add_argument(
        "--min-stocks", type=int, default=MIN_STOCKS,
        help=f"Minimum stocks per model (default: {MIN_STOCKS})",
    )
    parser.add_argument(
        "--max-stocks", type=int, default=MAX_STOCKS,
        help=f"Maximum stocks per model (default: {MAX_STOCKS})",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs per model (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL_MINUTES,
        help=f"Price sampling interval in minutes (default: {DEFAULT_INTERVAL_MINUTES})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    farm = ModelFarm(
        db_password=args.db_password,
        db_host=args.db_host,
        db_user=args.db_user,
        db_name=args.db_name,
        num_models=args.num_models,
        num_generations=args.num_generations,
        min_stocks=args.min_stocks,
        max_stocks=args.max_stocks,
        interval_minutes=args.interval,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    farm.run()


if __name__ == "__main__":
    main()
