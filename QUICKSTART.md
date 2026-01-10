# Quick Start Guide

## Setup (First Time Only)

1. **Install Python dependencies**:
   ```bash
   cd ~/pytink
   pip install -r requirements.txt
   ```

2. **Verify MySQL database** is running with:
   - Database: `tinker`
   - Host: `localhost`
   - Port: `3306`
   - User: `tinker`
   - Password: (provided via `--db-password` argument)

## Option 1: Run via Command Line (Default - Recommended)

```bash
# Default settings (20 stocks, 30-min intervals, 10 epochs, batch size 64, sequence length 16)
python train_model.py --db-password YOUR_PASSWORD
```

Or customize parameters:

```bash
python train_model.py \
  --db-password YOUR_PASSWORD \
  --num-stocks 50 \
  --interval 30 \
  --sequence-length 8 \
  --batch-size 128 \
  --epochs 20 \
  --learning-rate 0.0005
```

## Understanding the Output

### Vocabulary Analysis
- **Total words generated**: Number of unique stock movement sequences
- **Unique words**: Vocabulary size - how many different patterns exist
- **Vocabulary coverage**: Percentage of data that consists of unique patterns

### Training Metrics
- **Loss**: Should decrease during training
- **Accuracy**: Percentage of correct predictions on the dataset
- **Perplexity**: exp(loss) - lower is better

### Example Training Output
```
Epoch 1/10, Batch 10/45, Loss: 2.3456
Epoch 1/10 - Average Loss: 2.3200
...
Final Loss: 0.8542
Accuracy: 0.6234
Perplexity: 2.3496
```

## Data Flow

```
MySQL Database
    ↓
StockDatabase (fetch quotes)
    ↓
Raw price data (prices & timestamps)
    ↓
PriceProcessor (calculate deltas & encode)
    ↓
"Words" (encoded sequences like "acgaeb")
    ↓
StockWordDataset (create (input, label) pairs)
    ↓
StockTransformerModel (train & predict)
```

## Key Concepts

### Delta Encoding
Maps percentage price changes to letters:
```
Price Change    →    Letter
-1.0% or less  →    'a'
-0.5%          →    'b'
-0.1%          →    'c'
 0.0%          →    'd'
+0.1%          →    'e'
+0.5%          →    'f'
+1.0% or more  →    'g'
```

### Word Generation
Each "word" represents price movements across a portfolio:
- Word length = number of stocks
- Each character = one stock's price change
- Sequence of words = time series of portfolio movements

### Model Task
Given a sequence of 4 words, predict the next word (multi-step ahead prediction).

## Configuration Parameters

Pass via command line arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| --db-password | (required) | Database password |
| --num-stocks | 20 | Number of random stocks to analyze |
| --interval | 30 | Minutes between consecutive "words" |
| --sequence-length | 16 | Number of words for model input |
| --batch-size | 64 | Training batch size |
| --epochs | 10 | Number of training iterations |
| --learning-rate | 1e-5 | Adam optimizer learning rate |

## Troubleshooting

### "No words generated"
- Check that quote data exists for selected stocks
- Verify timestamps are aligned across stocks
- Try a longer date range or different stocks

### "Dataset is empty"
- Too many words may be needed for the sequence_length
- Reduce sequence_length or get more historical data

### Database connection error
- Ensure MySQL is running: `sudo service mysql status`
- Verify database `tinker` exists
- Check user credentials in database.py

### Out of memory
- Reduce batch_size (e.g., 16 or 8)
- Use fewer stocks (e.g., 5)
- Reduce hidden_size in model configuration

### Slow training
- Reduce num_epochs
- Use smaller hidden_size
- Reduce num_layers

## Next Steps

1. **Experiment with hyperparameters** - try different learning rates, model sizes
2. **Analyze predictions** - examine what patterns the model learns
3. **Add more stocks** - how does vocabulary size change?
4. **Try different intervals** - hourly or daily predictions
5. **Implement predictions** - use trained model to predict future prices
6. **Visualize results** - plot training curves, word distributions
7. **Save/load models** - persist trained models for reuse

## Files Generated

- **Model weights**: Can be saved to disk (modify train_model.py)
- **Vocabulary**: Save vocab dict for inference
- **Training history**: Loss and accuracy over time
- **Predictions**: Model outputs on test data

See `README.md` for detailed documentation.
