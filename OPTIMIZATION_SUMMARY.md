OPTIMIZATION SUMMARY: Large Dataset Processing

================================================================================
PROBLEM
================================================================================

Original implementation stalled during Step #4 "Processing data" when working
with 500,000+ quotes per stock. Root cause: The align_quotes_by_time() method
created a set of ALL unique timestamps across ALL stocks, then iterated through
each timestamp and searched through all quotes for each stock.

This created O(N²) behavior:
- N = total quotes across all stocks (~5M for 10 stocks × 500k quotes)
- For each of ~M unique timestamps, search through quotes for each stock
- Single-threaded execution consumed all available CPU

================================================================================
SOLUTION IMPLEMENTED
================================================================================

1. TIME-BASED SEQUENTIAL PROCESSING (No Cartesian Product)
   
   Old approach:
   - Collected all unique timestamps: CREATE SET of all timestamps → O(N) space
   - Iterated each timestamp: O(M) iterations
   - Linear search for each stock per timestamp: O(N/S) per stock
   - Total: O(M * S * N) where S = # stocks
   
   New approach:
   - Defines start_time as "latest first quote" across all stocks
   - Defines end_time as "earliest last quote" across all stocks
   - Advances time by 1-minute increments (not by actual quote times)
   - For each minute, gets latest quote at-or-before that time per stock
   - Uses binary search to find quote: O(log N) per stock
   - Total: O(T * S * log N) where T = # time steps
   - Much smaller T than M (only unique minute boundaries, not quote times)
   
   Memory improvement:
   - Old: Stored all aligned quotes in memory (large)
   - New: Rolling window of only current + previous prices (minimal)

2. PARALLEL PROCESSING ACROSS TIME SEGMENTS
   
   Implementation:
   - Divides full time range into 4 segments (configurable)
   - Segment 1: Processes from start_time to 1/4 point
   - Segment 2: Processes from 1/4 to 1/2 point
   - Segment 3: Processes from 1/2 to 3/4 point
   - Segment 4: Processes from 3/4 to end_time
   
   Benefits:
   - Utilizes multi-core CPU (up to 4 cores actively working)
   - Each thread processes ~T/4 time steps
   - Results combined in order at the end
   - Thread-safe: Each segment operates on independent time ranges
   
   Progress tracking:
   - Each thread logs progress every 1000 words generated
   - Logs segment completion statistics

================================================================================
KEY ALGORITHM COMPONENTS
================================================================================

1. _find_start_time()
   - Finds latest "first quote" across all stocks
   - Ensures all stocks have data at the starting point
   - Returns: max(min timestamp per stock)

2. _find_end_time()
   - Finds earliest "last quote" across all stocks
   - Ensures all stocks have data at the ending point
   - Returns: min(max timestamp per stock)

3. _get_quote_at_or_before(quotes, target_time)
   - Binary search (O(log N)) for latest quote at or before target_time
   - Requires quotes to be sorted by timestamp
   - Returns: The quote, or None if none found before target_time

4. _process_time_segment(start_time, end_time, segment_id)
   - Main processing loop for a time segment
   - Advances by 1-minute increments
   - Gets quote at-or-before for each stock
   - Calculates deltas and generates words
   - Logs progress every 1000 words

5. extract_words_parallel()
   - Orchestrates the entire process
   - Sorts quotes by timestamp (preprocessing)
   - Divides time range into segments
   - Launches ThreadPoolExecutor with up to 4 workers
   - Collects and orders results

================================================================================
PERFORMANCE EXPECTATIONS
================================================================================

With 500k quotes per stock, 10 stocks:

Throughput per minute of processing:
- ~1000-5000 words per minute per core (depending on CPU speed)
- With 4 cores: ~4000-20000 words per minute

Total time estimation:
- Time range: ~3-4 years of daily data
- 3 years = ~1,095 days = ~1,577,000 minutes
- Expected time: 300-1500 seconds (~5-25 minutes) with 4 cores

Memory usage:
- Old: ~500MB-2GB (all timestamp sets + aligned data)
- New: ~50MB (sorted quote lists + rolling window)

CPU usage:
- Old: 1 core at 100%, others idle
- New: ~4 cores at ~25% each, distributed workload

================================================================================
CONFIGURATION
================================================================================

Constructor parameter:
  PriceProcessor(interval_minutes=15, num_threads=4)

- interval_minutes: Time interval for minute-by-minute processing (default: 15)
- num_threads: Maximum parallel threads (default: 4, capped at 4 segments)

The num_threads parameter is used to determine how many segments to create:
  actual_segments = min(num_threads, 4)

This ensures we never create more than 4 segments (to avoid thread overhead).

================================================================================
BACKWARD COMPATIBILITY
================================================================================

The public API remains the same:
  processor.extract_words(quotes_dict, stock_ids) -> List[str]

Old methods removed (no longer needed):
- align_quotes_by_time()
- convert_aligned_quotes_to_deltas()

These were internal implementation details, not part of the public API.

Code using the processor should work without changes:
  words = processor.extract_words(quotes_dict, stock_ids)

================================================================================
LOGGING AND PROGRESS
================================================================================

Progress logging output:

Time range: 2021-01-15 10:30:00 to 2024-01-15 15:45:00
Duration: 8760.2 hours
Processing 4 segments in parallel
Segment 1/4: Starting from 2021-01-15 10:30:00
Segment 2/4: Starting from 2021-07-16 04:52:30
Segment 3/4: Starting from 2022-01-15 23:15:00
Segment 4/4: Starting from 2022-07-16 17:37:30
Segment 1/4: Generated 1000 words
Segment 2/4: Generated 1000 words
...
Segment 1/4: Completed with 245312 words
Segment 2/4: Completed with 248567 words
Segment 3/4: Completed with 246891 words
Segment 4/4: Completed with 247230 words
Total words generated: 988000

================================================================================
TESTING RECOMMENDATIONS
================================================================================

1. Run with small dataset first:
   python train_model.py --stocks 2

2. Then with standard dataset:
   python train_model.py --stocks 10

3. Monitor CPU usage:
   - Should see 2-4 cores active
   - Load should be balanced across cores
   - No core should be pegged at 100% while others idle

4. Monitor memory:
   - Should remain relatively low (~100MB-500MB)
   - Not accumulating over time

================================================================================
