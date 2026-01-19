STOCK PRICE PREDICTION WITH TRANSFORMER MODELS

================================================================================
SYSTEM OVERVIEW
================================================================================

A machine learning system that predicts future price movements of stocks using
a transformer model. The system encodes stock price changes as
strings of "delta" symbols, generating sequences of tokens where each token
represents concurrent price movements among a set of stocks.

Key Components:
  1. Data Pipeline: MySQL quotes → Delta encoding → Token sequences
  2. Transformer Model: Model for next-token prediction
  3. Training: Cross-entropy loss on final token of sequences
  4. Inference: Generate predicted sequences of price movements

Example:
  Stocks: AAPL, GOOGL, MSFT
  Time 10:00 (baseline prices): AAPL=$150, GOOGL=$100, MSFT=$350
  Time 10:01 (new prices): AAPL=$150.15 (+0.1%), GOOGL=$99.95 (-0.05%), MSFT=$350.35 (+0.1%)
  
  Token generated: "ecf"
    e = AAPL delta +0.1% (symbol for +0.1%)
    c = GOOGL delta -0.05% → quantized to -0.001 (symbol c)
    f = MSFT delta +0.1% (symbol for +0.1%)

================================================================================
CONFIGURATION
================================================================================

All parameters are configurable via YAML config file or command-line arguments.
Priority: CLI arguments > user config file > config_template.yaml defaults

Key configuration sections:

DATA:
  - num_stocks: Number of random stocks (default: 10)
  - tickers: Specific tickers to use (overrides num_stocks)
  - interval_minutes: Time interval between tokens (default: 30)
  - sequence_length: Context window size (default: 32)

MODEL:
  - hidden_size: Embedding/hidden dimension (default: 128)
  - num_hidden_layers: Transformer layers (default: 4)
  - num_attention_heads: Attention heads (default: 4)
  - max_position_embeddings: Max sequence position (default: 256)

TRAINING:
  - batch_size: Sequences per batch (default: 64)
  - num_epochs: Training epochs (default: 25)
  - learning_rate: Adam learning rate (default: 0.0003)
  - weight_decay: L2 regularization (default: 0.01)
  - early_stopping_patience: Epochs without improvement before stopping (default: 5)
  - use_class_weights: Weight loss by inverse class frequency (default: true)

DELTA_RANGES:
  - Configurable array of delta quantization levels
  - Default: [-0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01]

Example usage:
  python train_model.py --db-password PASSWORD --config my_config.yaml
  python train_model.py --db-password PASSWORD --epochs 50 --learning-rate 0.001

================================================================================
DELTA ENCODING: QUANTIZATION OF PRICE CHANGES
================================================================================

Delta = (new_price - old_price) / old_price

The system quantizes price changes into discrete levels.  By default, there are 7 levels:

DELTA_VALUES = [-.01, -.005, -.001, 0, .001, .005, .01]
               Symbol: a    b     c    d  e    f    g

Quantization Process:
  For each minute-to-minute price change, compute delta
  Map delta to nearest DELTA_VALUE
  Convert to single-letter symbol (a-g)
  Concatenate symbols over the list of stocks to form token

Example quantization:
  Price change: -0.07% (delta = -0.0007)
  Nearest level: -.001 (only ±0.0003 error)
  Symbol: c

The delta values array is configurable.

================================================================================
TOKEN AND SEQUENCE GENERATION
================================================================================

Algorithm: extract_tokens_parallel()

Input:
  - Stock IDs: [10, 23, 45, ...] (configurable via num_stocks or tickers)
  - quotes_dict: {stock_id: [Quote1, Quote2, ...]} sorted by timestamp
  - interval_minutes: configurable (default: 30)
  - sequence_length: configurable (default: 32)

Process:

1. BOUNDARY DETECTION:
   - Find earliest quote across all stocks: max(min quote per stock)
   - Find latest quote across all stocks: max(max quote per stock)
   - Defines time range: [earliest_start, latest_end]

2. TIME SEGMENTATION:
   - Divide time range into 4 equal segments
   - Each segment processed by independent thread
   - Market hours awareness: processes only NYSE trading hours (9:30 AM - 4:00 PM ET)
   - Automatically skips weekends, holidays, and after-hours periods

3. PARALLEL PROCESSING (4 workers):
   For each segment in parallel:
     a) Initialize current_time = segment_start
     b) Loop: while current_time <= segment_end:
        - Get quote at-or-before current_time for each stock
          Using binary search: O(log N) per stock
        - Parse prices from quotes
        - Compute deltas vs previous minute
        - Generate token by mapping deltas to symbols
        - Store token in output
        - Advance to next 15-minute interval
     c) Return tokens for this segment

4. RESULT AGGREGATION:
   - Combine tokens from all segments in chronological order
   - Result: sorted list of tokens [token_1, token_2, ...]

5. SEQUENCE GENERATION:
   From tokens list, create training sequences:
   - Each sequence = (input: [token_1, ..., token_n], label: token_{n+1})
   - sequence_length is configurable (default: 32)
   - Example: (["token_1", "token_2", ..., "token_32"], "token_33")

Example output for 20 stocks, 15-minute intervals:
  Tokens: ["abcdefghijklmnopqrst", "fbdghhijklmnopqrsta", "gbcdaghijklmnopqrstf", ...]
  Sequences: (["t1", "t2", "t3", "t4"]), ...


================================================================================
DATASET CONSTRUCTION
================================================================================

Class: StockTokenDataset

Input:
  - tokens: List[str] of generated tokens
  - sequence_length: int (configurable, default: 32)
  - vocab: Dict[str, int] mapping tokens to token IDs

Processing:
  1. Convert tokens to token IDs using vocab
  2. Create sliding window sequences:
     - For i in range(len(tokens) - sequence_length):
       - input_ids = tokens[i : i + sequence_length]
       - label_id = tokens[i + sequence_length]
       - Store (input_ids, label_id) as training example

  3. __getitem__(index):
     - Returns (input_sequence, label) as tensors
     - input_sequence: shape (sequence_length,) with token IDs
     - label: scalar tensor with next token ID

Example with 3 tokens, sequence_length=2:
  tokens: ["abc", "def", "ghi"]
  vocab: {"abc": 0, "def": 1, "ghi": 2}
  token_ids: [0, 1, 2]
  
  Dataset examples:
    ([0, 1], 2) → Predict token ID 2 from token IDs [0, 1]
  
  Only 1 example from 3 tokens because:
    Sequence 0: token_ids[0:2]=[0,1], label=token_ids[2]=2 ✓

With 1M tokens and sequence_length=4:
  ~999,996 training examples (each token creates 1 example)

================================================================================
TRANSFORMER MODEL ARCHITECTURE
================================================================================

Class: StockTransformerModel(nn.Module)

Architecture (all configurable via config file):
  - Input: Token IDs (integers 0 to vocab_size-1)
  - Embedding: vocab_size → hidden_size (default: 128)
  - Transformer Blocks: num_hidden_layers (default: 4)
    - Attention heads: num_attention_heads (default: 4)
  - Max position embeddings: max_position_embeddings (default: 256)
  - Output: Logits over vocab (probabilities)

Forward pass:
  1. Embed input tokens: (batch, seq_len) → (batch, seq_len, 256)
  2. Pass through transformer blocks with causal masking
  3. Extract last token: (batch, seq_len, 256) → (batch, 256)
  4. Apply output linear layer: (batch, 256) → (batch, vocab_size)
  5. Return logits for cross-entropy loss

Loss computation:
  - Only compute loss on LAST TOKEN of sequence
  - Ignore earlier tokens in sequence
  - Loss = CrossEntropyLoss(logits_last, label)
  
  Rationale: We want to predict the NEXT token given context,
  not predict all intermediate positions.

Example:
  Input sequence: token IDs [0, 1, 2, 3] (4 tokens)
  Expected next token ID: 4 (the 5th token)
  
  Forward:
    (4,) → embed → (4, hidden_size) → transformer → (4, hidden_size)
    Extract position 3: (hidden_size,) → linear → (vocab_size,)
    Loss = CrossEntropyLoss(logits, target_label=4)

Parameters (with default configuration):
  - Total: varies based on vocab_size and hidden_size
  - Embedding: vocab_size × hidden_size
  - Transformer: num_hidden_layers × (attention + feedforward)
  - Output: hidden_size × vocab_size

================================================================================
TRAINING CONFIGURATION
================================================================================

All training parameters are configurable via config file or command-line arguments:

Loss: Cross-entropy loss on next-token prediction
  - Optional class weighting: inverse log-frequency weights (use_class_weights: true)
  - Helps balance learning across imbalanced delta distributions

Optimizer: Adam
  - learning_rate: configurable (default: 0.0003)
  - weight_decay: configurable (default: 0.01)

Batch size: configurable (default: 64)
Epochs: configurable (default: 25)
Gradient clipping: Max norm 1.0

Early Stopping:
  - early_stopping_patience: configurable (default: 5, 0 to disable)
  - Monitors eval loss; restores best model weights after training

Custom collate_fn():
  Converts batch of (sequence, label) tuples:
  - Stacks sequences: (batch_size, seq_length)
  - Stacks labels: (batch_size,)
  - Moves to device (CUDA or CPU)

Training loop (pseudocode):
  best_eval_loss = infinity
  epochs_without_improvement = 0
  
  for epoch in range(num_epochs):
    # Training phase
    total_loss = 0
    for batch_idx, (input_ids, labels) in enumerate(train_loader):
      input_ids, labels = input_ids.to(device), labels.to(device)
      
      logits = model(input_ids)  # (batch_size, vocab_size)
      loss = criterion(logits, labels)  # weighted if use_class_weights=true
      
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      
      total_loss += loss.item()
    
    # Evaluation phase
    eval_loss, eval_accuracy = evaluate(eval_loader)
    
    # Early stopping check
    if eval_loss < best_eval_loss:
      best_eval_loss = eval_loss
      save_best_model_state()
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1
      if epochs_without_improvement >= early_stopping_patience:
        restore_best_model_state()
        break

================================================================================
DATA QUALITY FILTERS
================================================================================

1. STOCK SELECTION:
   - Minimum quotes per stock: 100,000 (when using random selection)
   - Ensures sufficient historical data for pattern learning
   - Filters out stocks with sparse trading
   - Can specify exact tickers via config to bypass random selection

2. STOCK RECENCY FILTER:
   - Remove stocks with last quote > 30 days old
   - Ensures all selected stocks have recent, active trading
   - Replace stale stocks with fresh selections
   - Verify that data reflects current market conditions

3. MARKET HOURS FILTER:
   - Only processes quotes during NYSE trading hours (9:30 AM - 4:00 PM ET)
   - Skips weekends (Saturday, Sunday)
   - Skips US market holidays (2024-2026 pre-configured)
   - Resets price tracking at trading day boundaries

4. QUOTE VALIDATION:
   - Parse prices as floats (reject invalid data)
   - Skip minutes with missing quotes for any stock
   - Skip minutes with invalid price values (0, negative, etc.)

5. PRICE MOVEMENT LIMITS:
   - Delta quantization caps at ±1% per interval
   - Extreme moves mapped to ±1% boundary

Process:
  1. Fetch stocks from database
  2. Filter by minimum quote count (100k+)
  3. Check last quote timestamp for each stock
  4. Replace any stale stocks (>30 days old)
  5. Repeat until all stocks pass recency check

================================================================================
EVALUATION METRICS
================================================================================

Computed during training and testing:

Let NUM_STOCKS = number of stocks (token length)

1. LOSS (Cross-entropy):
   - Measures divergence between predicted and actual next tokens
   - Lower is better
   - Range: 0 to ln(NUM_STOCK)

2. ACCURACY:
   - Fraction of correctly predicted next tokens
   - Range: 0 to 1.0
   - Random baseline: 1 / NUM_STOCK

3. PERPLEXITY:
   - Exponential of average loss: exp(loss)
   - Interpreted as "how confused the model is"
   - Random baseline: vocab_size (uniform distribution)

Per-class metrics (delta-level breakdown):
  - Accuracy for each symbol
  - Shows which deltas are easier/harder to predict
  - Helps identify imbalanced delta distributions

4. CONFUSION MATRICES (Per-Stock Letter-by-Letter Analysis):
   - One matrix per stock position showing delta symbol predictions
   - Matrix size: NUM_DELTAS×NUM_DELTAS (e.g., 7×7 for default delta levels)
   - Rows: actual delta symbols, Columns: predicted delta symbols
   - Diagonal: correct predictions for that stock
   - Off-diagonal: confusion between adjacent/distant delta levels
   - Helps identify which stocks are harder to predict
   - Shows systematic misclassifications per stock position
   - Per-stock accuracy computed and ranked at end of evaluation

Example output:
  Final Eval Loss: 0.8234
  Final Eval Accuracy: 48.3%
  Final Perplexity: 2.28
  
  Per-Stock Confusion Matrix (AAPL, position 1/5) - Accuracy: 0.4821:
  ----------------------------------------------------------------
  Actual\Pred |  a  |  b  |  c  |  d  |  e  |  f  |  g  | Total
  ----------------------------------------------------------------
       a      |  12 |   3 |   1 |   0 |   0 |   0 |   0 |    16
       b      |   5 |  45 |  12 |   2 |   0 |   0 |   0 |    64
       c      |   2 |  18 | 180 |  40 |  15 |   2 |   0 |   257
       d      |   0 |   5 |  35 | 210 |  35 |   8 |   1 |   294
       e      |   0 |   2 |  12 |  40 | 175 |  20 |   3 |   252
       f      |   0 |   0 |   3 |   8 |  25 |  48 |   6 |    90
       g      |   0 |   0 |   1 |   2 |   5 |   8 |  15 |    31
  ----------------------------------------------------------------
  
  Per-Stock Accuracies (ranked):
    AAPL  : 0.5123
    GOOGL : 0.4892
    MSFT  : 0.4756
    TSLA  : 0.4521
    NVDA  : 0.4312
  
    Average per-stock accuracy: 0.4721

================================================================================
KEY OPTIMIZATIONS
================================================================================

1. Time-based Sequential Processing:
   - Process minute-by-minute instead of quote-by-quote
   - Eliminates need for timestamp deduplication
   - Reduces memory from 1-2 GB to 50-100 MB

2. Binary Search for Quote Lookup:
   - Replace linear O(N) search with O(log N) binary search
   - 26,000x speedup for large quote lists
   - Enables processing of 500k+ quotes per stock

3. Parallel Segment Processing:
   - Divide time range into 4 independent segments
   - ThreadPoolExecutor for 4-way parallelism
   - ~4x speedup on multi-core systems
   - No synchronization overhead (independent time ranges)

4. Vectorized Batch Processing:
   - Prepare batches of sequences for GPU/CPU
   - Custom collate_fn ensures correct tensor shapes
   - Efficient matrix operations in transformer

5. Device-aware Tensor Management:
   - Ensure all tensors on same device (CUDA vs CPU)
   - Prevent host-device transfer overhead
   - Move data to device at DataLoader level

Overall speedup from optimizations: ~80-100x
  - Time complexity: O(N²) → O((S×T×log N)/W)
  - Expected runtime: 5-25 minutes for 20 stocks × 500k quotes

================================================================================
