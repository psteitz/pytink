# Usage Examples

This document provides practical examples of how to use the stock prediction system.

## Example 1: Command-Line Quick Analysis

```bash
# Run with defaults (20 stocks, 30-min intervals, 10 epochs, batch size 64, sequence length 16)
python train_model.py --db-password YOUR_PASSWORD

# Output:
# ✓ Successfully connected to database
# ✓ Database contains 100 stocks
# Selected 20 random stocks:
#   AAPL: Apple Inc (ID: 1)
#   MSFT: Microsoft Corp (ID: 15)
#   ...
# Generated 5234 words
# Unique words: 847
# ...
# Epoch 1/10 - Average Loss: 2.3200
# Epoch 2/10 - Average Loss: 1.8450
# ...
# Final Loss: 0.8542
# Accuracy: 0.6234
# Perplexity: 2.3496
```

## Example 2: Custom Configuration
```bash
# Use specific parameters
python train_model.py \
    --db-password YOUR_PASSWORD \
    --num-stocks 15 \
    --interval 30 \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 5e-4 \
    --sequence-length 6

# Explanation:
# - 15 stocks instead of 20
# - 30-minute intervals
# - 20 epochs instead of 10
# - Smaller batch size (more frequent updates)
# - Higher learning rate (faster convergence)
# - Shorter context (6 words instead of default 16)
```

## Example 3: Programmatic Usage

```python
import sys
sys.path.insert(0, 'src')

from database import StockDatabase
from processor import PriceProcessor
from model import StockWordDataset, StockTransformerModel
import torch

# Connect to database (password is required)
db = StockDatabase(password='YOUR_PASSWORD')
db.connect()

# Get specific stocks
stocks = db.get_random_stocks(count=5)
stock_ids = [s['id'] for s in stocks]
print(f"Selected: {[s['ticker'] for s in stocks]}")

# Fetch quotes
quotes = db.get_quotes_for_stocks(stock_ids)

# Process data
processor = PriceProcessor(interval_minutes=15)
words = processor.extract_words(quotes, stock_ids)
print(f"Generated {len(words)} words")

# Analyze
unique_count, unique_words = processor.count_unique_words(words)
print(f"Vocabulary size: {unique_count}")

# Create dataset
vocab = {w: i for i, w in enumerate(sorted(unique_words))}
dataset = StockWordDataset(words, vocab, sequence_length=4)

# Initialize model
model = StockTransformerModel(vocab_size=len(vocab))

# Get predictions
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32)

model.eval()
with torch.no_grad():
    for input_ids, labels in loader:
        predictions = model.predict(input_ids)
        # predictions shape: (batch_size,)
        print(f"Predicted tokens: {predictions}")

db.close()
```

## Example 4: Analyzing Word Patterns

```python
from collections import Counter

# After generating words...
word_freq = Counter(words)

# Top 10 most common patterns
print("Most common price movement patterns:")
for word, count in word_freq.most_common(10):
    pct = count / len(words) * 100
    print(f"  '{word}': {count} times ({pct:.2f}%)")

# Word length (should be constant = number of stocks)
word_length = len(words[0]) if words else 0
print(f"Word length: {word_length} chars (one per stock)")

# Vocabulary statistics
print(f"Total words: {len(words)}")
print(f"Unique words: {len(word_freq)}")
print(f"Vocabulary coverage: {len(word_freq) / len(words) * 100:.2f}%")
```

## Example 5: Custom Delta Ranges

You can customize the delta encoding ranges by specifying them in a YAML config file.

**Create a custom config file** (`my_config.yaml`):
```yaml
# Custom delta ranges - wider range with 9 levels (±10%)
delta_ranges:
  - -0.10    # a: -10%
  - -0.05    # b: -5%
  - -0.02    # c: -2%
  - -0.01    # d: -1%
  -  0.0     # e: 0%
  -  0.01    # f: +1%
  -  0.02    # g: +2%
  -  0.05    # h: +5%
  -  0.10    # i: +10%

data:
  num_stocks: 20
  interval_minutes: 30
```

**Run with custom config:**
```bash
python train_model.py --db-password YOUR_PASSWORD --config my_config.yaml
```

**Or use custom ranges programmatically:**
```python
from src.processor import PriceProcessor

# Create processor with custom delta ranges (wider ±5% range)
custom_deltas = [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05]
processor = PriceProcessor(interval_minutes=30, delta_values=custom_deltas)

# Now delta_to_symbol uses your custom ranges
delta = 0.03  # 3%
symbol = processor.delta_to_symbol(delta)
print(f"3% change → '{symbol}'")  # → 'g' (closest to +2%)

# The default ranges are:
# a: -1%, b: -0.5%, c: -0.1%, d: 0%
# e: +0.1%, f: +0.5%, g: +1%
```

## Example 6: Model Training with Custom Configuration

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Create data loader
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Setup training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = StockTransformerModel(vocab_size=len(vocab), device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Training loop
num_epochs = 5
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    
    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model.get_model()(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
```

## Example 7: Evaluate on Specific Stocks

```python
# Get quotes for specific tickers
all_stocks = db.get_all_stocks()
target_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Find stock IDs
target_stocks = [s for s in all_stocks if s['ticker'] in target_tickers]
stock_ids = [s['id'] for s in target_stocks]

# Process only these stocks
quotes = db.get_quotes_for_stocks(stock_ids)
words = processor.extract_words(quotes, stock_ids)

print(f"Generated {len(words)} words for {target_tickers}")

# Create and train model on these specific stocks
vocab = {w: i for i, w in enumerate(sorted(set(words)))}
dataset = StockWordDataset(words, vocab)
# ... continue with training
```

## Example 8: Batch Experiments

```bash
# Run multiple experiments with different configurations

echo "=== Quick Test (Small model, few epochs) ==="
python train_model.py --db-password YOUR_PASSWORD --num-stocks 5 --epochs 2

echo "=== Standard Configuration ==="
python train_model.py --db-password YOUR_PASSWORD --num-stocks 10 --epochs 10

echo "=== Large Scale (More stocks, longer training) ==="
python train_model.py --db-password YOUR_PASSWORD --num-stocks 20 --epochs 20

echo "=== Fine-tuned Learning ==="
python train_model.py --db-password YOUR_PASSWORD --learning-rate 2e-4 --batch-size 16

# Compare results in output logs
```

## Example 9: Save and Load Model

```python
# After training...

# Save model weights
torch.save(model.get_model().state_dict(), 'stock_model.pt')
print("Model saved to stock_model.pt")

# Save vocabulary
import json
with open('vocab.json', 'w') as f:
    json.dump(vocab, f)
print("Vocabulary saved to vocab.json")

# Later, load and use model
from transformers import AutoModelForCausalLM
from model import StockTransformerModel

# Load vocab
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# Create and load model
loaded_model = StockTransformerModel(vocab_size=len(vocab))
loaded_model.get_model().load_state_dict(torch.load('stock_model.pt'))
loaded_model.eval()

# Use for predictions
with torch.no_grad():
    predictions = loaded_model.predict(input_ids)
```

## Example 10: Visualize Training Progress

```python
import matplotlib.pyplot as plt

# After training with history dict...
plt.figure(figsize=(12, 5))
plt.plot(training_history['loss'])
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Over Batches')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('training_loss.png')
plt.show()

# Plot by epoch
epochs_data = {}
for epoch, loss in zip(training_history['epoch'], training_history['loss']):
    if epoch not in epochs_data:
        epochs_data[epoch] = []
    epochs_data[epoch].append(loss)

epoch_nums = sorted(epochs_data.keys())
epoch_losses = [sum(epochs_data[e]) / len(epochs_data[e]) for e in epoch_nums]

plt.figure(figsize=(10, 5))
plt.plot(epoch_nums, epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Loss Per Epoch')
plt.grid(True, alpha=0.3)
plt.savefig('epoch_loss.png')
plt.show()
```

## Troubleshooting Examples

### Issue: "No words generated"

```python
# Debug: Check data alignment
aligned = processor.align_quotes_by_time(quotes, stock_ids)
print(f"Aligned timestamps: {len(aligned)}")
print(f"First alignment: {aligned[0] if aligned else 'None'}")

# Check if stocks have overlapping time ranges
for stock_id in stock_ids:
    if stock_id in quotes:
        timestamps = [q['timestamp'] for q in quotes[stock_id]]
        print(f"Stock {stock_id}: {min(timestamps)} to {max(timestamps)}")
```

### Issue: "Dataset is empty"

```python
# Check word count
print(f"Total words: {len(words)}")
print(f"Sequence length: {sequence_length}")
print(f"Possible sequences: {len(words) - sequence_length}")

# Solution: reduce sequence length if needed
if len(words) - sequence_length < 100:
    sequence_length = max(1, len(words) - 100)
    print(f"Reduced sequence length to {sequence_length}")
```

### Issue: Out of memory

```python
# Reduce batch size
batch_size = 8  # was 64

# Reduce data size
stocks = db.get_random_stocks(count=5)  # was 20

# Use shorter sequences
sequence_length = 8  # was 16
```

---

## Example 12: Evaluating a Trained Model

After training a model with `save_model: true`, evaluate it on recent data:

**Basic inference:**
```bash
# Evaluate on last 3 months of data (default)
python inference.py --db-password YOUR_PASSWORD \
    --model-dir models/AAPL-GOOGL-MSFT/20260101_120000/
```

**Custom time range:**
```bash
# Evaluate on last 6 months
python inference.py --db-password YOUR_PASSWORD \
    --model-dir models/AAPL-GOOGL-MSFT/20260101_120000/ \
    --months 6
```

**With custom batch size:**
```bash
python inference.py --db-password YOUR_PASSWORD \
    --model-dir models/AAPL-GOOGL-MSFT/20260101_120000/ \
    --months 3 --batch-size 128
```

**Output example:**
```
Loading model from models/AAPL-GOOGL-MSFT/20260101_120000/
Stocks: AAPL, GOOGL, MSFT
Evaluation period: last 3 months (2025-10-01 to 2026-01-02)

Results:
  Accuracy: 45.23%
  Loss: 1.856
  Perplexity: 6.40
  Total samples: 1,234

Per-stock confusion matrix...
```

**Using inference programmatically:**
```python
from inference import load_model_config, load_model, evaluate_model

# Load saved model
model_dir = 'models/AAPL-GOOGL-MSFT/20260101_120000/'
config = load_model_config(model_dir)
model, symbol_to_id = load_model(model_dir, config)

# Evaluate on your own data
# ... fetch and process quotes ...
# results = evaluate_model(model, dataset, symbol_to_id)
```

---

For more information, see README.md, QUICKSTART.md, and PROJECT_SUMMARY.md
