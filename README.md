# Stock Price Prediction with Transformers

A transformer-based model that treats stock price movements as a language modeling problem, predicting future price patterns from historical sequences.

## Motivation

This project explores the question: **what if we treat stock price movements like words in a language?**

Just as transformer models learn to predict the next word in a sentence by understanding context and patterns, this system learns to predict the next "word" of price movements across a portfolio of stocks. Each "word" encodes simultaneous price changes across multiple stocks at a given time interval.

Stock prices don't move in isolation. The idea here is to see if an attention-based approach can work to learn relationships between stock movements over time.  Models focus on a vector of stocks.  The length is configurable, defaulting to 20 randomly selected high-volume stocks. Each (configurable) time increment, changes are recorded for each stock in the vector.  The changes are quantized into (configurable) bins (e.g, [-.01, -.005, -.0001, 0, .0001, .005, .01]) which are mapped to letters.  The letters are concatentated to form "words" and the transformer model is trained to predict the next word in the sequence.

## How It Works

1. **Quantize price changes** into discrete symbols (a-g representing -1% to +1%)
2. **Create "words"** by concatenating symbols for all stocks at each time interval
3. **Build sequences** of consecutive words (like sentences)
4. **Train a transformer** to predict the next word given previous words

The model learns which price movement patterns tend to follow other patterns—essentially learning the "grammar" of market movements.

## Project Overview

This project trains a transformer model to predict the next sequence of stock price changes ("words") given a history of previous sequences. Stock price changes are encoded as letters (a-g) representing different percentage change ranges.

### Delta Encoding

Price changes are mapped to letters as follows:
- `a`: -1.0% (-.01)
- `b`: -0.5% (-.005)
- `c`: -0.1% (-.001)
- `d`: 0.0% (0)
- `e`: +0.1% (+.001)
- `f`: +0.5% (+.005)
- `g`: +1.0% (+.01)

### Example

A "word" like `acgaeb` with 6 stocks means:
- Stock 1: `a` (-1%)
- Stock 2: `c` (-0.1%)
- Stock 3: `g` (+1%)
- Stock 4: `a` (-1%)
- Stock 5: `e` (+0.1%)
- Stock 6: `b` (-0.5%)

## Project Structure

```
pytink/
├── train_model.py           # Main CLI entry point
├── src/
│   ├── database.py          # MySQL database interface
│   ├── processor.py         # Price processing and delta encoding
│   ├── model.py             # PyTorch model and dataset classes
│   └── analysis.py          # Visualization utilities
├── tests/                   # pytest test suite (76 tests)
│   ├── test_database.py     # Database tests (7 tests)
│   ├── test_processor.py    # Processor tests (35 tests)
│   ├── test_model.py        # Model tests (22 tests)
│   └── test_integration.py  # Integration tests (12 tests)
├── models/                  # Saved model files (git-ignored)
├── logs/                    # Training logs (git-ignored)
├── config_template.yaml     # Configuration template
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.8+
- MySQL 5.7+
- See `requirements.txt` for Python packages

### Database Setup

The project expects a local MySQL database with:
- **Database name**: `tinker`
- **Port**: 3306
- **User**: `tinker`
- **Password**: Provided via `--db-password` command-line argument

**Tables**:
- `stocks`: Contains `id` (INT), `ticker` (VARCHAR), `name` (VARCHAR)
- `quotes`: Contains `price` (VARCHAR), `timestamp` (DATETIME), `stock` (INT foreign key)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pytink
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
pytest tests/ -q
```

## Quick Start

```bash
# See all options
python train_model.py -h

# Run with defaults (20 stocks, 10 epochs)
python train_model.py --db-password YOUR_PASSWORD

# Custom configuration
python train_model.py --db-password YOUR_PASSWORD --stocks 10 --epochs 5 --interval 15
```

## Usage

### Running the Analysis

Run the model training script:

```bash
python train_model.py --db-password YOUR_PASSWORD --stocks 10 --interval 15 --sequence-length 8 --batch-size 64
```

The script performs the following workflow:

1. **Connect to Database**: Establish MySQL connection and verify tables
2. **Select Stocks**: Randomly select specified number of stocks with sufficient data
3. **Fetch Historical Data**: Retrieve price quotes for selected stocks
4. **Process Price Data**: Convert price time series into delta sequences (market hours only)
5. **Generate Words**: Encode deltas as letter sequences (a-g)
6. **Analyze Patterns**: Display top 10 most common price movement patterns
7. **Prepare Dataset**: Create PyTorch dataset with input-output pairs
8. **Train Model**: Run training loop with AdamW optimizer
9. **Evaluate**: Calculate loss, accuracy, and perplexity metrics
10. **Save Model**: Save trained model to `models/<TICKERS>_model.pt`

### Market Hours Awareness

The processor automatically:
- Skips weekends (Saturday/Sunday)
- Skips US market holidays
- Only processes data during market hours (9:30 AM - 4:00 PM ET)
- Resets price baselines at each market open to avoid cross-day artifacts

## Configuration

Parameters can be adjusted via command line or `config_template.yaml`:

- **--db-password**: Database password (required)
- **--config**: Path to YAML configuration file
- **--stocks**: Number of stocks to analyze (default: 20)
- **--interval**: Price sampling interval in minutes (default: 30)
- **--sequence-length**: Context window for model input (default: 16)
- **--batch-size**: Training batch size (default: 64)
- **--learning-rate**: AdamW optimizer learning rate (default: 1e-5)
- **--epochs**: Number of training epochs (default: 10)
- **--save-model**: Save trained model to disk (default: True)

### Custom Delta Ranges

You can customize the delta encoding ranges in your config file:

```yaml
delta_ranges:
  - -0.05    # a: -5%
  - -0.02    # b: -2%
  - -0.01    # c: -1%
  -  0.0     # d: 0%
  -  0.01    # e: +1%
  -  0.02    # f: +2%
  -  0.05    # g: +5%
```

### Model Architecture

The transformer model uses:
- **Vocabulary Size**: Number of unique words in the dataset
- **Hidden Size**: 256 dimensions
- **Layers**: 6 transformer layers
- **Attention Heads**: 8
- **Position Embeddings**: Up to 256 tokens

## Module Documentation

### `database.py`

`StockDatabase` class provides:
- `connect()`: Establish MySQL connection
- `get_all_stocks()`: Retrieve all stocks
- `get_random_stocks(count)`: Get random stock sample
- `get_quotes_for_stock(stock_id)`: Fetch quotes for one stock
- `get_quotes_for_stocks(stock_ids)`: Fetch quotes for multiple stocks

### `processor.py`

`PriceProcessor` class provides:
- `parse_price(price_str)`: Convert price strings to floats
- `calculate_delta(old_price, new_price)`: Calculate percentage change
- `delta_to_symbol(delta)`: Map delta to letter
- `symbol_to_delta(symbol)`: Map letter back to delta
- `align_quotes_by_time(quotes_dict, stock_ids)`: Align quotes from multiple stocks
- `extract_words(quotes_dict, stock_ids)`: Generate words from price data
- `count_unique_words(words)`: Count vocabulary size

### `model.py`

`StockTransformerModel` class provides:
- `forward(input_ids, labels)`: Forward pass with optional loss computation
- `predict(input_ids)`: Generate predictions
- `train()` / `eval()`: Set model mode

`StockWordDataset` class:
- PyTorch Dataset for word sequences
- Returns (input_ids, label) pairs for training

## Performance Metrics

The analysis script calculates:
- **Loss**: Cross-entropy loss on the dataset
- **Accuracy**: Percentage of correct predictions
- **Perplexity**: Exp(loss), a common NLP metric

## Notes

- With 10 stocks and 7 delta levels, the maximum possible vocabulary is 7^10 ≈ 282 million words, but actual data typically contains far fewer unique words
- The model learns patterns in how stock prices change together
- Database connectivity is required; ensure MySQL is running before starting
- Models are saved to `models/<TICKERS>_model.pt` by default

## Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_processor.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Future Enhancements

- Implement train/validation/test splits
- Try different model architectures (increased layers, attention heads)
- Add regularization techniques (dropout, layer normalization)
- Generate longer sequences (multi-step ahead predictions)
- Analyze prediction patterns for trading signals
- Add support for other markets (extended hours, international exchanges)

## Documentation

- **QUICKSTART.md**: 5-minute getting started guide
- **EXAMPLES.md**: Detailed usage examples
- **ALGORITHM_DETAILS.md**: Technical deep-dive into the encoding scheme
- **PROJECT_SUMMARY.md**: Architecture overview

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.