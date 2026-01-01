STOCK PRICE PREDICTION WITH TRANSFORMER MODELS

================================================================================
SYSTEM OVERVIEW
================================================================================

A machine learning system that predicts future price movements of stocks using
a GPT-2 style transformer model. The system encodes stock price changes as
discrete "delta" symbols, generating sequences of words where each word
represents concurrent price movements across multiple stocks.

Key Components:
  1. Data Pipeline: MySQL quotes → Delta encoding → Word sequences
  2. Transformer Model: GPT-2 style architecture for next-token prediction
  3. Training: Cross-entropy loss on final token of sequences
  4. Inference: Generate predicted sequences of price movements

Example:
  Stocks: AAPL, GOOGL, MSFT
  Time 10:00 (baseline prices): AAPL=$150, GOOGL=$100, MSFT=$350
  Time 10:01 (new prices): AAPL=$150.15 (+0.1%), GOOGL=$99.95 (-0.05%), MSFT=$350.35 (+0.1%)
  
  Word generated: "ecf"
    e = AAPL delta +0.1% (symbol for +0.1%)
    c = GOOGL delta -0.05% → quantized to -0.001 (symbol c)
    f = MSFT delta +0.1% (symbol for +0.1%)

================================================================================
DELTA ENCODING: 7-LEVEL QUANTIZATION
================================================================================

Delta = (new_price - old_price) / old_price

The system quantizes price changes into 7 discrete levels:

DELTA_VALUES = [-.01, -.005, -.001, 0, .001, .005, .01]
               Symbol: a    b     c    d  e    f    g

Quantization Process:
  For each minute-to-minute price change, compute delta
  Map delta to nearest DELTA_VALUE
  Convert to single-letter symbol (a-g)
  Concatenate symbols across all stocks

Example quantization:
  Price change: -0.07% (delta = -0.0007)
  Nearest level: -.001 (only ±0.0003 error)
  Symbol: c

Rationale for 7 levels (±1% with 0.1% steps):
  - Range: ±1% captures normal minute-to-minute movements
  - Finer granularity: 0.1% steps for precise classification
  - More balanced distribution: distributes price moves across categories
  - Prevents model from exploiting 0% dominance

Characteristics:
  - Vocabulary size: 7 symbols (a-g)
  - Per-stock information: 1 symbol per stock per minute
  - Multi-stock encoding: S symbols per minute (S = number of stocks)
  - Word length: S characters (e.g., "fce" for 3 stocks)

================================================================================
WORD AND SEQUENCE GENERATION
================================================================================

Algorithm: extract_words_parallel()

Input:
  - Stock IDs: [10, 23, 45, ...] (typically 20 stocks)
  - quotes_dict: {stock_id: [Quote1, Quote2, ...]} sorted by timestamp
  - interval_minutes: 15 (default)
  - sequence_length: 4 (sequences of 4 words)

Process:

1. BOUNDARY DETECTION:
   - Find earliest quote across all stocks: max(min quote per stock)
   - Find latest quote across all stocks: max(max quote per stock)
   - Defines time range: [earliest_start, latest_end]

2. TIME SEGMENTATION:
   - Divide time range into 4 equal segments
   - Each segment processed by independent thread

3. PARALLEL PROCESSING (4 workers):
   For each segment in parallel:
     a) Initialize current_time = segment_start
     b) Loop: while current_time <= segment_end:
        - Get quote at-or-before current_time for each stock
          Using binary search: O(log N) per stock
        - Parse prices from quotes
        - Compute deltas vs previous minute
        - Generate word by mapping deltas to symbols
        - Store word in output
        - Advance to next 15-minute interval
     c) Return words for this segment

4. RESULT AGGREGATION:
   - Combine words from all segments in chronological order
   - Result: sorted list of words [word_1, word_2, ...]

5. SEQUENCE GENERATION:
   From words list, create training sequences:
   - Each sequence = (input: [word_1, ..., word_n], label: word_{n+1})
   - Default sequence_length = 8
   - Example: (["word_1", "word_2", "word_3", "word_4"], "word_5")

Example output for 20 stocks, 15-minute intervals:
  Words: ["abcdefghijklmnopqrst", "fbdghhijklmnopqrsta", "gbcdaghijklmnopqrstf", ...]
  Sequences: (["w1", "w2", "w3", "w4"], "w5"), ...

================================================================================
QUOTE LOOKUP OPTIMIZATION: BINARY SEARCH
================================================================================

Method: _get_quote_at_or_before(quotes: List[Dict], target_time: datetime)

Quotes are pre-sorted by timestamp. For each minute in time range, we need
the latest quote available at or before that minute.

Naive approach: Linear search O(N)
  result = None
  for quote in quotes:
    if quote.timestamp <= target_time:
      result = quote  # Keep updating
  return result
  
  With 500k quotes: 500,000 comparisons per minute
  For 1.5M minutes: 750 billion comparisons

Binary search approach: O(log N)
  left = 0
  right = len(quotes) - 1
  result = None
  
  while left <= right:
    mid = (left + right) // 2
    if quotes[mid].timestamp <= target_time:
      result = quotes[mid]  # Found candidate
      left = mid + 1         # Search for newer
    else:
      right = mid - 1        # Search for older
  
  return result
  
  With 500k quotes: ~19 comparisons per minute max (log2(500000))
  For 1.5M minutes: 28.5 million comparisons

Speedup: 750B / 28.5M ≈ 26,000x faster

================================================================================
DATASET CONSTRUCTION
================================================================================

Class: StockWordDataset

Input:
  - words: List[str] of generated words
  - sequence_length: int (default 8)
  - vocab: Dict[str, int] mapping words to token IDs

Processing:
  1. Convert words to token IDs using vocab
  2. Create sliding window sequences:
     - For i in range(len(tokens) - sequence_length):
       - input_ids = tokens[i : i + sequence_length]
       - label_id = tokens[i + sequence_length]
       - Store (input_ids, label_id) as training example

  3. __getitem__(index):
     - Returns (input_sequence, label) as tensors
     - input_sequence: shape (sequence_length,) with token IDs
     - label: scalar tensor with next token ID

Example with 3 words, sequence_length=2:
  words: ["abc", "def", "ghi"]
  vocab: {"abc": 0, "def": 1, "ghi": 2}
  tokens: [0, 1, 2]
  
  Dataset examples:
    ([0, 1], 2) → Predict token 2 from tokens [0, 1]
  
  Only 1 example from 3 words because:
    Sequence 0: tokens[0:2]=[0,1], label=tokens[2]=2 ✓

With 1M words and sequence_length=4:
  ~999,996 training examples (each word creates 1 example)

================================================================================
TRANSFORMER MODEL ARCHITECTURE
================================================================================

Class: StockTransformerModel(nn.Module)

Architecture:
  - Input: Token IDs (integers 0 to vocab_size-1)
  - Embedding: vocab_size → embedding_dim (256)
  - Transformer Blocks: 6 layers
    - Attention heads: 8
    - Feed-forward dim: 1024
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
  
  Rationale: We want to predict the NEXT word given context,
  not predict all intermediate positions.

Example:
  Input sequence: tokens [0, 1, 2, 3] (4 words)
  Expected next token: 4 (the 5th word)
  
  Forward:
    (4,) → embed → (4, 256) → transformer → (4, 256)
    Extract position 3: (256,) → linear → (vocab_size,)
    Loss = CrossEntropyLoss(logits, target_label=4)

Parameters:
  - Total: ~1-1.5M trainable parameters
  - Embedding: vocab_size × 256
  - Transformer: 6 layers × (attention + feedforward)
  - Output: 256 × vocab_size

================================================================================
TRAINING CONFIGURATION
================================================================================

Loss: Cross-entropy loss on next-token prediction
Optimizer: Adam (learning rate: 1e-5)
Batch size: 64 sequences per batch
Epochs: 10 (default)
Gradient clipping: Max norm 1.0

Custom collate_fn():
  Converts batch of (sequence, label) tuples:
  - Stacks sequences: (batch_size, seq_length)
  - Stacks labels: (batch_size,)
  - Moves to device (CUDA or CPU)

Training loop (pseudocode):
  for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
      input_ids, labels = input_ids.to(device), labels.to(device)
      
      logits = model(input_ids)  # (batch_size, 7)
      loss = criterion(logits, labels)
      
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      
      total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

================================================================================
DATA QUALITY FILTERS
================================================================================

1. STOCK SELECTION:
   - Minimum quotes per stock: 100,000
   - Ensures sufficient historical data for pattern learning
   - Filters out stocks with sparse trading

2. STOCK RECENCY FILTER:
   - Remove stocks with last quote > 30 days old
   - Ensures all selected stocks have recent, active trading
   - Automatically replaces stale stocks with fresh selections
   - Validates that data reflects current market conditions

3. QUOTE VALIDATION:
   - Parse prices as floats (reject invalid data)
   - Skip minutes with missing quotes for any stock
   - Skip minutes with invalid price values (0, negative, etc.)

4. PRICE MOVEMENT LIMITS:
   - Delta quantization caps at ±1% per interval
   - Extreme moves (stock halts, gaps) mapped to ±1% boundary
   - Prevents outliers from dominating training

Process:
  1. Fetch random stocks from database
  2. Filter by minimum quote count (100k+)
  3. Check last quote timestamp for each stock
  4. Replace any stale stocks (>30 days old)
  5. Repeat until all stocks pass recency check

================================================================================
EVALUATION METRICS
================================================================================

Computed during training and testing:

1. LOSS (Cross-entropy):
   - Measures divergence between predicted and actual next tokens
   - Lower is better
   - Range: 0 to ln(7) ≈ 1.95 (if uniform random)
   - Good performance: < 1.0

2. ACCURACY:
   - Fraction of correctly predicted next tokens
   - Range: 0 to 1.0
   - Random baseline (7 classes): 1/7 ≈ 14.3%
   - Good performance: > 25%

3. PERPLEXITY:
   - Exponential of average loss: exp(loss)
   - Interpreted as "how confused the model is"
   - Random baseline: vocab_size (uniform distribution)
   - Good performance: < 3

Per-class metrics (delta-level breakdown):
  - Accuracy for each symbol (a-g)
  - Shows which deltas are easier/harder to predict
  - Helps identify imbalanced delta distributions

Example output:
  Loss: 0.8234
  Accuracy: 48.3%
  Perplexity: 2.28
  
  Per-symbol accuracy:
    a (-1%): 52.1%
    b (-0.5%): 49.8%
    c (-0.1%): 46.2%
    d (0%): 55.3%
    e (+0.1%): 43.7%
    f (+0.5%): 47.9%
    g (+1%): 48.1%

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
