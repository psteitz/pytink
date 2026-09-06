#!/usr/bin/env python3
"""Simulation experiments evaluating trained stock-prediction models as trading signals.

All experiments address the question: *given a model trained by train_model, what would
happen if I used its predictions to decide when to buy stocks and hold them for a fixed
period?*

By default the CLI evaluates 10 known-good ticker sets against **3 independently
random trading days** drawn from the configurable date window (2025-06-01 – 2026-03-31).
This produces 30 independent experiment runs and prints a grand-summary table of total
gain, total invested, and overall return % across all runs.

The experiments here are a stepping stone toward a live alert application that polls
stock prices every 15 minutes and notifies the user when the model recommends buying any
of the stocks in its portfolio.

Known good ticker sets (``KNOWN_GOOD_MODELS``, used by default):

    F-MMM-NFLX-MSFT-XOMA
    TSLA-NFLX-LIFE-TSN-MRNA
    F-VGLT-PCTY-SFM-LIFE
    FB-ZNGA-JPM-MA-SHOP
    UBER-SPY-MRK-NVDA-JNJ
    TSN-DIS-SQQQ-VGLT-PFE
    MSFT-VTI-NOK-QCOM-NFLX
    SWN-KO-LIFE-SABR-M
    SWN-ZNGA-QQQ-JPM-RIVN
    UBER-DIS-KO-SHOP-JPM-FB-ZNGA

Usage examples:

  # Default: 10 known-good models, 3 random dates each
  python -m pytink.experiments --db-password PASSWORD

  # Evaluate a single specific ticker set on a specific date
  python -m pytink.experiments --db-password PASSWORD \\
      --tickers TSLA-NFLX-LIFE-TSN-MRNA --end-date 2026-02-15

  # More training epochs, bigger target purchase
  python -m pytink.experiments --db-password PASSWORD \\
      --epochs 10 --buy-dollars 10000

  # Predict 5 steps ahead instead of the default 3
  python -m pytink.experiments --db-password PASSWORD --predict-steps 5

  # Use 5 random dates instead of 3
  python -m pytink.experiments --db-password PASSWORD --num-dates 5
"""

import argparse
import logging
import random
import sys
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

import torch

from pytink.database import StockDatabase
from pytink.inference import predict_next_tokens
from pytink.model import StockTransformerModel
from pytink.processor import PriceProcessor, US_MARKET_HOLIDAYS
from pytink.train_model import prepare_data, train_and_evaluate

logger = logging.getLogger(__name__)

# ── Known good models ─────────────────────────────────────────────────────────

KNOWN_GOOD_MODELS: List[str] = [
    "F-MMM-NFLX-MSFT-XOMA",
    "TSLA-NFLX-LIFE-TSN-MRNA",
    "F-VGLT-PCTY-SFM-LIFE",
    "FB-ZNGA-JPM-MA-SHOP",
    "UBER-SPY-MRK-NVDA-JNJ",
    "TSN-DIS-SQQQ-VGLT-PFE",
    "MSFT-VTI-NOK-QCOM-NFLX",
    "SWN-KO-LIFE-SABR-M",
    "SWN-ZNGA-QQQ-JPM-RIVN",
    "UBER-DIS-KO-SHOP-JPM-FB-ZNGA",
]

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_TICKER_SET: str = "F-MMM-NFLX-MSFT-XOMA"

# Simulation "present day" range.  Random dates are drawn from this window when
# --end-date is not supplied.
SIM_DATE_MIN: date = date(2025, 6, 1)
SIM_DATE_MAX: date = date(2026, 8, 31)

# Buy decision parameters
BUY_TARGET_DOLLARS: float = 5_000.0
DEFAULT_PREDICT_STEPS: int = 3

# Number of random trading days to sample per ticker set when no --end-date is given
DEFAULT_NUM_DATES: int = 3

# Data / model architecture — mirror farming.py defaults for fair comparison
DEFAULT_INTERVAL_MINUTES: int = 30
DEFAULT_CONTEXT_WINDOW_SIZE: int = 256
DEFAULT_EPOCHS: int = 5
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LEARNING_RATE: float = 3e-4
DEFAULT_WEIGHT_DECAY: float = 0.01
DEFAULT_EARLY_STOPPING_PATIENCE: int = 3
HIDDEN_SIZE: int = 128
NUM_HIDDEN_LAYERS: int = 4
NUM_ATTENTION_HEADS: int = 4
MAX_POSITION_EMBEDDINGS: int = 256


# ── Calendar helpers ──────────────────────────────────────────────────────────

def random_end_date(
    min_date: date = SIM_DATE_MIN,
    max_date: date = SIM_DATE_MAX,
) -> date:
    """Return a uniformly random date in [min_date, max_date]."""
    span = (max_date - min_date).days
    return min_date + timedelta(days=random.randint(0, span))


def is_trading_day(d: date) -> bool:
    """Return True when *d* is a NYSE trading day (Mon–Fri, not a holiday)."""
    return d.weekday() < 5 and d not in US_MARKET_HOLIDAYS


def next_trading_day(d: date) -> date:
    """Return the first trading day strictly after *d*."""
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def select_random_trading_dates(
    n: int,
    min_date: date = SIM_DATE_MIN,
    max_date: date = SIM_DATE_MAX,
) -> List[date]:
    """Return *n* distinct random trading days in [min_date, max_date].

    Dates are sampled without replacement and returned in ascending order.
    Raises ValueError if fewer than *n* trading days exist in the range.

    Args:
        n: Number of dates to select.
        min_date: Earliest possible date (inclusive).
        max_date: Latest possible date (inclusive).

    Returns:
        Sorted list of *n* distinct trading days.
    """
    candidates = [
        min_date + timedelta(days=i)
        for i in range((max_date - min_date).days + 1)
        if is_trading_day(min_date + timedelta(days=i))
    ]
    if len(candidates) < n:
        raise ValueError(
            f"Only {len(candidates)} trading days exist between {min_date} and "
            f"{max_date}, but {n} were requested."
        )
    return sorted(random.sample(candidates, n))


# ── Price lookup ──────────────────────────────────────────────────────────────

def get_closing_price(quotes: List[Dict], target_date: date) -> Optional[float]:
    """Return the last quoted price on *target_date*, or None if none exists.

    Args:
        quotes: List of quote dicts with 'price' and 'timestamp' keys, sorted
            by timestamp ascending.
        target_date: The calendar date whose closing price is desired.

    Returns:
        The last price recorded on that date, or None.
    """
    day_start = datetime.combine(target_date, time(0, 0, 0))
    day_end = datetime.combine(target_date, time(23, 59, 59))

    last_price: Optional[float] = None
    for quote in quotes:
        ts = quote['timestamp']
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if day_start <= ts <= day_end:
            last_price = float(quote['price'])
    return last_price


# ── Position sizing ───────────────────────────────────────────────────────────

def compute_shares(price: float, target_dollars: float = BUY_TARGET_DOLLARS) -> int:
    """Return the integer share count whose cost is closest to *target_dollars*.

    Rounds to the nearest whole number of shares, with a minimum of 1.

    Args:
        price: Current price per share in dollars.
        target_dollars: Desired position size (default: BUY_TARGET_DOLLARS).

    Returns:
        Number of whole shares to purchase (>= 1).
    """
    if price <= 0:
        return 1
    return max(1, round(target_dollars / price))


# ── Buy decision logic ────────────────────────────────────────────────────────

def make_buy_decisions(
    predicted_words: List[Optional[str]],
    tickers: List[str],
    processor: PriceProcessor,
) -> List[str]:
    """Return the subset of *tickers* that meet the buy criteria.

    A ticker at word-position *i* is a buy candidate when:

    1. The **first** predicted word encodes one of the **two highest** positive
       delta values for position *i* (e.g. +0.005 or +0.01 for the default
       quantizer — letters 'f' or 'g').
    2. **None** of the predicted words encodes a delta below −0.001 for
       position *i* (letters corresponding to deltas ≤ −0.005 are vetoes).

    Args:
        predicted_words: List of predicted word strings (one per prediction
            step).  An entry of None means the model produced an unknown token.
        tickers: Ordered list of ticker symbols matching word character positions.
        processor: PriceProcessor used during training; supplies delta_values.

    Returns:
        List of ticker symbols to buy, in the same order as *tickers*.
    """
    if not predicted_words or predicted_words[0] is None:
        return []

    delta_values = processor.delta_values
    # The two highest letters encode the top-2 positive deltas (buy signal)
    buy_letters = frozenset(
        chr(ord('a') + i)
        for i in range(max(0, len(delta_values) - 2), len(delta_values))
    )
    # Letters whose delta is strictly below -0.001 act as vetoes
    veto_letters = frozenset(
        chr(ord('a') + i)
        for i, d in enumerate(delta_values)
        if d < -0.001
    )

    to_buy = []
    for stock_idx, ticker in enumerate(tickers):
        first_word = predicted_words[0]
        if stock_idx >= len(first_word):
            continue
        if first_word[stock_idx] not in buy_letters:
            continue
        # Veto check across all predicted tokens
        vetoed = any(
            word is not None
            and stock_idx < len(word)
            and word[stock_idx] in veto_letters
            for word in predicted_words
        )
        if not vetoed:
            to_buy.append(ticker)

    return to_buy


# ── Core experiment ───────────────────────────────────────────────────────────

def run_buy_hold_experiment(
    db: StockDatabase,
    tickers: List[str],
    end_date: date,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    context_window_size: int = DEFAULT_CONTEXT_WINDOW_SIZE,
    predict_steps: int = DEFAULT_PREDICT_STEPS,
    buy_target_dollars: float = BUY_TARGET_DOLLARS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    device: Optional[str] = None,
) -> Dict:
    """Run a single buy-and-hold simulation experiment.

    Trains a model on quotes up to *end_date*, predicts the next *predict_steps*
    tokens using the tail of the training sequence as context, applies the buy
    decision rule, and simulates holding each purchased stock for one trading day.

    The experiment answers: *if we trusted this model to tell us when to buy,
    how much money would we have made or lost?*

    Args:
        db: Connected StockDatabase instance.
        tickers: Ordered list of ticker symbols to include in the model.
        end_date: Simulation "present day" — training data cutoff and buy date.
        interval_minutes: Price-sampling interval for PriceProcessor.
        context_window_size: Input sequence length for the transformer.
        predict_steps: Number of future tokens to predict autoregressively.
        buy_target_dollars: Target spend per stock purchase (rounded to whole shares).
        epochs: Maximum training epochs (early stopping may reduce this).
        batch_size: Training batch size.
        learning_rate: Adam optimizer learning rate.
        weight_decay: Adam optimizer weight decay.
        early_stopping_patience: Stop training after this many epochs without
            improvement (0 disables early stopping).
        device: Torch device string; auto-detects CUDA when None.

    Returns:
        Dict with the following keys:

        - ``end_date`` (date): The simulation cutoff date.
        - ``tickers`` (list[str]): Ticker symbols in model order.
        - ``train_loss`` (float): Final training-set eval loss.
        - ``train_accuracy`` (float): Final training-set eval accuracy.
        - ``predicted_words`` (list): Raw predicted token strings.
        - ``buy_decisions`` (list[str]): Tickers selected for purchase.
        - ``purchases`` (list[dict]): Per-stock detail dicts with keys
          ``ticker``, ``shares``, ``buy_price``, ``sell_price``,
          ``invested``, ``gain``.
        - ``total_invested`` (float): Total dollars spent across all buys.
        - ``total_gain`` (float): Total dollars gained (or lost) after selling.
        - ``return_pct`` (float): Percentage return on invested capital.

    Raises:
        ValueError: When any ticker is not found in the database.
        RuntimeError: When the quote data is insufficient to train a model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Resolve stock IDs in caller-specified order ───────────────────────────
    stocks = db.get_stocks_by_tickers(tickers)
    found = {s['ticker']: s for s in stocks}
    missing = [t for t in tickers if t not in found]
    if missing:
        raise ValueError(f"Tickers not found in database: {missing}")

    ordered_stocks = [found[t] for t in tickers]
    stock_ids = [s['id'] for s in ordered_stocks]

    logger.info(
        "Experiment: tickers=%s  end_date=%s  device=%s",
        tickers, end_date, device,
    )

    # ── Fetch training quotes (all data up to and including end_date) ─────────
    training_cutoff = datetime.combine(end_date, time(23, 59, 59))
    logger.info("Fetching training quotes up to %s...", training_cutoff)
    quotes_dict = db.get_quotes_for_stocks(stock_ids, end_date=training_cutoff)

    for sid, quotes in quotes_dict.items():
        ticker = found[next(t for t in tickers if found[t]['id'] == sid)]['ticker']
        logger.info("  %s: %d quotes", ticker, len(quotes))

    # ── Prepare data ──────────────────────────────────────────────────────────
    prepared = prepare_data(
        quotes_dict=quotes_dict,
        stock_ids=stock_ids,
        interval_minutes=interval_minutes,
        context_window_size=context_window_size,
        batch_size=batch_size,
        min_words=context_window_size + 10,
        min_sequences=10,
    )
    if prepared is None:
        raise RuntimeError(
            "Data preparation failed — not enough quotes for "
            f"tickers={tickers} up to end_date={end_date}."
        )

    processor = prepared.processor
    words = prepared.words
    vocab = prepared.vocab
    logger.info("Prepared %d words, vocab size %d", len(words), len(vocab))

    # ── Build and train model ─────────────────────────────────────────────────
    model = StockTransformerModel(
        vocab_size=len(vocab),
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        device=device,
    )

    train_loss, train_accuracy = train_and_evaluate(
        model=model,
        train_loader=prepared.train_loader,
        eval_loader=prepared.eval_loader,
        eval_dataset_len=len(prepared.eval_subset),
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        device=device,
    )
    logger.info(
        "Training complete — loss=%.4f  accuracy=%.4f",
        train_loss, train_accuracy,
    )

    # ── Predict next tokens using tail of training words as seed ──────────────
    predicted_words = predict_next_tokens(
        model, words, vocab, context_window_size, predict_steps, device
    )
    logger.info("Predicted %d token(s): %s", predict_steps, predicted_words)

    # ── Apply buy decision rule ───────────────────────────────────────────────
    buy_tickers = make_buy_decisions(predicted_words, tickers, processor)
    logger.info("Buy decisions: %s", buy_tickers if buy_tickers else "none")

    if not buy_tickers:
        return {
            'end_date': end_date,
            'tickers': tickers,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'predicted_words': predicted_words,
            'buy_decisions': [],
            'purchases': [],
            'total_invested': 0.0,
            'total_gain': 0.0,
            'return_pct': 0.0,
        }

    # ── Fetch buy-day and sell-day quotes ─────────────────────────────────────
    sell_date = next_trading_day(end_date)
    buy_stock_ids = [found[t]['id'] for t in buy_tickers]
    logger.info(
        "Fetching buy/sell quotes for %s on %s (buy) and %s (sell)...",
        buy_tickers, end_date, sell_date,
    )
    day_quotes = db.get_quotes_for_stocks(
        buy_stock_ids,
        start_date=datetime.combine(end_date, time(0, 0, 0)),
        end_date=datetime.combine(sell_date, time(23, 59, 59)),
    )

    # ── Simulate purchases and compute P&L ────────────────────────────────────
    purchases = []
    total_invested = 0.0
    total_gain = 0.0

    for ticker in buy_tickers:
        stock_id = found[ticker]['id']
        quotes = day_quotes.get(stock_id, [])

        buy_price = get_closing_price(quotes, end_date)
        sell_price = get_closing_price(quotes, sell_date)

        if buy_price is None:
            logger.warning(
                "No buy-day quote for %s on %s — skipping this position.", ticker, end_date
            )
            continue
        if sell_price is None:
            logger.warning(
                "No sell-day quote for %s on %s — skipping this position.", ticker, sell_date
            )
            continue

        shares = compute_shares(buy_price, buy_target_dollars)
        invested = shares * buy_price
        proceeds = shares * sell_price
        gain = proceeds - invested

        purchases.append({
            'ticker': ticker,
            'shares': shares,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'invested': invested,
            'gain': gain,
        })
        total_invested += invested
        total_gain += gain

        logger.info(
            "  %s: %d shares @ $%.2f → $%.2f | gain $%+.2f",
            ticker, shares, buy_price, sell_price, gain,
        )

    return_pct = (total_gain / total_invested * 100.0) if total_invested > 0 else 0.0

    return {
        'end_date': end_date,
        'tickers': tickers,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'predicted_words': predicted_words,
        'buy_decisions': buy_tickers,
        'purchases': purchases,
        'total_invested': total_invested,
        'total_gain': total_gain,
        'return_pct': return_pct,
    }


# ── Result display ────────────────────────────────────────────────────────────

def _print_grand_summary(results: List[Dict]) -> None:
    """Log a condensed table summarising multiple experiment results."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("GRAND SUMMARY  (%d experiment(s))", len(results))
    logger.info("=" * 80)
    logger.info(
        "%-30s  %-12s  %10s  %10s  %8s",
        "Tickers", "Date", "Invested", "Gain", "Return%",
    )
    logger.info("-" * 80)

    grand_invested = 0.0
    grand_gain = 0.0
    for r in results:
        ticker_str = "-".join(r["tickers"])
        if len(ticker_str) > 30:
            ticker_str = ticker_str[:27] + "..."
        logger.info(
            "%-30s  %-12s  %10.2f  %+10.2f  %+7.2f%%",
            ticker_str,
            str(r["end_date"]),
            r["total_invested"],
            r["total_gain"],
            r["return_pct"],
        )
        grand_invested += r["total_invested"]
        grand_gain += r["total_gain"]

    overall_return = (grand_gain / grand_invested * 100.0) if grand_invested > 0 else 0.0
    logger.info("-" * 80)
    logger.info(
        "%-44s  %10.2f  %+10.2f  %+7.2f%%",
        "TOTAL",
        grand_invested,
        grand_gain,
        overall_return,
    )
    logger.info("=" * 80)


def _print_result(result: Dict) -> None:
    """Log a human-readable summary of an experiment result."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPERIMENT RESULT")
    logger.info("=" * 60)
    logger.info("Simulation date : %s", result['end_date'])
    logger.info("Tickers         : %s", ', '.join(result['tickers']))
    logger.info(
        "Model           : loss=%.4f  accuracy=%.4f",
        result['train_loss'], result['train_accuracy'],
    )
    logger.info("Predicted tokens: %s", result['predicted_words'])
    logger.info(
        "Buy decisions   : %s",
        ', '.join(result['buy_decisions']) if result['buy_decisions'] else 'none',
    )
    logger.info("")

    if not result['purchases']:
        logger.info("No purchases executed.  Gain/loss: $0.00")
    else:
        logger.info("%-8s  %6s  %10s  %10s  %10s", "Ticker", "Shares", "Buy $", "Sell $", "Gain $")
        logger.info("-" * 50)
        for p in result['purchases']:
            logger.info(
                "%-8s  %6d  %10.2f  %10.2f  %+10.2f",
                p['ticker'], p['shares'], p['buy_price'], p['sell_price'], p['gain'],
            )
        logger.info("-" * 50)
        logger.info("Total invested : $%.2f", result['total_invested'])
        logger.info("Total gain/loss: $%+.2f", result['total_gain'])
        logger.info("Return         : %+.2f%%", result['return_pct'])

    logger.info("=" * 60)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for running the buy-and-hold experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description='Run a buy-and-hold simulation experiment using a trained model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--db-password', required=True, help='MySQL database password')
    parser.add_argument('--db-host', default='localhost', help='Database host')
    parser.add_argument('--db-user', default='tinker', help='Database user')
    parser.add_argument('--db-name', default='tinker', help='Database name')
    parser.add_argument(
        '--tickers',
        default=None,
        help=(
            'Dash-delimited ticker list (e.g. AAPL-GOOG-MSFT). '
            'When omitted all 10 KNOWN_GOOD_MODELS are evaluated.'
        ),
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help=(
            'Simulation cutoff date YYYY-MM-DD. '
            'When omitted, --num-dates random trading days are selected from '
            f'[{SIM_DATE_MIN}, {SIM_DATE_MAX}].'
        ),
    )
    parser.add_argument(
        '--num-dates',
        type=int,
        default=DEFAULT_NUM_DATES,
        help=(
            f'Number of random trading days to sample per ticker set '
            f'(default: {DEFAULT_NUM_DATES}). Ignored when --end-date is supplied.'
        ),
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help='Training epochs',
    )
    parser.add_argument(
        '---steps',
        type=inpredictt,
        default=DEFAULT_PREDICT_STEPS,
        help='Number of future tokens to predict',
    )
    parser.add_argument(
        '--buy-dollars',
        type=float,
        default=BUY_TARGET_DOLLARS,
        help='Target position size per stock in dollars',
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=DEFAULT_INTERVAL_MINUTES,
        help='Price-sampling interval in minutes',
    )
    parser.add_argument(
        '--context-window-size',
        type=int,
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
        help='Model input sequence length',
    )

    args = parser.parse_args()

    # ── Determine ticker sets to evaluate ────────────────────────────────────
    if args.tickers:
        ticker_sets = [[t.strip().upper() for t in args.tickers.split('-') if t.strip()]]
        if not ticker_sets[0]:
            logger.error("No tickers found in --tickers argument.")
            sys.exit(1)
    else:
        ticker_sets = [
            [t.strip() for t in model_str.split('-') if t.strip()]
            for model_str in KNOWN_GOOD_MODELS
        ]
        logger.info("No --tickers supplied; evaluating all %d known-good models.", len(ticker_sets))

    # ── Determine simulation dates ────────────────────────────────────────────
    if args.end_date:
        try:
            end_dates = [datetime.strptime(args.end_date, '%Y-%m-%d').date()]
        except ValueError:
            logger.error("Cannot parse --end-date '%s' as YYYY-MM-DD.", args.end_date)
            sys.exit(1)
    else:
        end_dates = select_random_trading_dates(args.num_dates)
        logger.info(
            "Selected %d random trading date(s): %s",
            len(end_dates),
            ", ".join(str(d) for d in end_dates),
        )

    db = StockDatabase(
        password=args.db_password,
        host=args.db_host,
        user=args.db_user,
        database=args.db_name,
    )
    db.connect()

    all_results: List[Dict] = []
    total_runs = len(ticker_sets) * len(end_dates)
    run_num = 0

    try:
        for tickers in ticker_sets:
            for end_date in end_dates:
                run_num += 1
                logger.info(
                    "\n[Run %d/%d]  tickers=%s  end_date=%s",
                    run_num, total_runs, tickers, end_date,
                )
                try:
                    result = run_buy_hold_experiment(
                        db=db,
                        tickers=tickers,
                        end_date=end_date,
                        interval_minutes=args.interval,
                        context_window_size=args.context_window_size,
                        predict_steps=args.predict_steps,
                        buy_target_dollars=args.buy_dollars,
                        epochs=args.epochs,
                    )
                    _print_result(result)
                    all_results.append(result)
                except Exception as exc:
                    logger.error(
                        "Run %d/%d failed (%s / %s): %s",
                        run_num, total_runs, '-'.join(tickers), end_date, exc,
                    )
    finally:
        db.close()

    if all_results:
        _print_grand_summary(all_results)


if __name__ == '__main__':
    main()
