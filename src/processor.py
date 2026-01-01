"""Data processing utilities for stock price deltas."""
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime, timedelta, time, date
import bisect
import numpy as np
import logging
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# US Eastern timezone for market hours
ET = ZoneInfo("America/New_York")

# US Market hours (Eastern Time)
MARKET_OPEN = time(9, 30)   # 9:30 AM ET
MARKET_CLOSE = time(16, 0)  # 4:00 PM ET

# US Market holidays (2020-2030) - dates when NYSE is closed
# This is a simplified list; could be extended or loaded from external source
US_MARKET_HOLIDAYS = {
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}


# Delta to character mapping
DELTA_VALUES = [-.01, -.005, -.001, 0, +.001, +.005, +.01]
DELTA_TO_CHAR = {delta: chr(ord('a') + i) for i, delta in enumerate(DELTA_VALUES)}
CHAR_TO_DELTA = {chr(ord('a') + i): delta for i, delta in enumerate(DELTA_VALUES)}


class PriceProcessor:
    """Processes stock price data and converts to deltas and words.
    
    Optimized for large datasets (500k+ quotes per stock) using:
    - Time-based sequential processing (no cartesian product of timestamps)
    - Parallel processing across time segments
    - Minimal memory footprint (rolling 15-minute window)
    - Market hours awareness (skips weekends, holidays, after-hours)
    """
    
    def __init__(self, interval_minutes: int = 15, num_threads: int = 4, delta_values: List[float] = None):
        """Initialize processor with time interval, thread count, and optional custom delta values.
        
        Args:
            interval_minutes: Time interval between words in minutes
            num_threads: Number of threads for parallel processing
            delta_values: Custom list of delta values for quantization (default: 7-level ±1%)
        """
        self.interval = timedelta(minutes=interval_minutes)
        
        # Use custom delta values if provided, otherwise use defaults
        if delta_values is not None:
            self.delta_values = sorted(delta_values)
        else:
            self.delta_values = DELTA_VALUES
        
        # Build character mappings dynamically based on delta_values
        self.delta_to_char = {delta: chr(ord('a') + i) for i, delta in enumerate(self.delta_values)}
        self.char_to_delta = {chr(ord('a') + i): delta for i, delta in enumerate(self.delta_values)}
        
        self.num_threads = num_threads
        self.words_lock = Lock()
    
    def _is_market_open(self, dt: datetime) -> bool:
        """Check if the market is open at the given datetime.
        
        Args:
            dt: Datetime to check (will be converted to ET if timezone-aware,
                otherwise assumed to be in ET)
        
        Returns:
            True if market is open (M-F, 9:30 AM - 4:00 PM ET, not a holiday)
        """
        # Convert to ET if timezone-aware, otherwise assume ET
        if dt.tzinfo is not None:
            dt_et = dt.astimezone(ET)
        else:
            dt_et = dt.replace(tzinfo=ET)
        
        # Check if weekend (Monday=0, Sunday=6)
        if dt_et.weekday() >= 5:
            return False
        
        # Check if holiday
        if dt_et.date() in US_MARKET_HOLIDAYS:
            return False
        
        # Check if within market hours
        current_time = dt_et.time()
        return MARKET_OPEN <= current_time < MARKET_CLOSE
    
    def _next_market_open(self, dt: datetime) -> datetime:
        """Get the next market open time after the given datetime.
        
        Args:
            dt: Starting datetime
        
        Returns:
            Datetime of next market open (9:30 AM ET on next trading day)
        """
        # Convert to ET
        if dt.tzinfo is not None:
            dt_et = dt.astimezone(ET)
        else:
            dt_et = dt.replace(tzinfo=ET)
        
        # Start with next day at market open
        next_day = dt_et.date() + timedelta(days=1)
        candidate = datetime.combine(next_day, MARKET_OPEN).replace(tzinfo=ET)
        
        # Skip weekends and holidays
        while candidate.weekday() >= 5 or candidate.date() in US_MARKET_HOLIDAYS:
            candidate += timedelta(days=1)
        
        # Return in same timezone format as input
        if dt.tzinfo is None:
            return candidate.replace(tzinfo=None)
        return candidate
    
    def parse_price(self, price_str: str) -> Optional[float]:
        """Parse price string to float."""
        try:
            return float(price_str)
        except (ValueError, TypeError):
            return None
    
    def calculate_delta(self, old_price: float, new_price: float) -> Optional[float]:
        """Calculate percentage change from old to new price."""
        if old_price == 0 or old_price is None or new_price is None:
            return None
        return (new_price - old_price) / old_price
    
    def delta_to_symbol(self, delta: float) -> Optional[str]:
        """Map a delta value to the closest symbol."""
        if delta is None:
            return None
        closest_delta = min(self.delta_values, key=lambda x: abs(x - delta))
        return self.delta_to_char[closest_delta]
    
    def symbol_to_delta(self, symbol: str) -> float:
        """Map a symbol back to its delta value."""
        return self.char_to_delta.get(symbol)
    
    def _parse_timestamp(self, ts) -> datetime:
        """Parse timestamp to datetime object."""
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)
        return ts
    
    def _get_quote_at_or_before(
        self,
        quotes: List[Dict],
        target_time: datetime,
        timestamps: List[datetime] = None
    ) -> Optional[Dict]:
        """Get the latest quote at or before target_time using binary search.
        
        Assumes quotes are sorted by timestamp.
        
        Args:
            quotes: List of quote dictionaries
            target_time: Target datetime to search for
            timestamps: Pre-parsed list of timestamps (for performance)
        """
        if not quotes:
            return None
        
        # Use pre-parsed timestamps if available, otherwise parse on the fly
        if timestamps is not None:
            # Use bisect for O(log N) search with C implementation
            idx = bisect.bisect_right(timestamps, target_time)
            if idx == 0:
                return None
            return quotes[idx - 1]
        else:
            # Fallback to manual binary search if timestamps not pre-parsed
            left, right = 0, len(quotes) - 1
            result = None
            
            while left <= right:
                mid = (left + right) // 2
                quote_time = self._parse_timestamp(quotes[mid]['timestamp'])
                
                if quote_time <= target_time:
                    result = quotes[mid]
                    left = mid + 1
                else:
                    right = mid - 1
            
            return result
    
    def _find_start_time(
        self,
        quotes_dict: Dict[int, List[Dict]],
        stock_ids: List[int]
    ) -> datetime:
        """Find the latest first quote timestamp across all stocks.
        
        This becomes the beginning of our data capture window.
        """
        max_first_time = None
        
        for stock_id in stock_ids:
            if stock_id not in quotes_dict or not quotes_dict[stock_id]:
                continue
            
            first_quote_time = self._parse_timestamp(quotes_dict[stock_id][0]['timestamp'])
            
            if max_first_time is None or first_quote_time > max_first_time:
                max_first_time = first_quote_time
        
        return max_first_time
    
    def _find_end_time(
        self,
        quotes_dict: Dict[int, List[Dict]],
        stock_ids: List[int]
    ) -> datetime:
        """Find the latest last quote timestamp across all stocks.
        
        This allows processing all available data. If a stock has fewer recent quotes,
        the quote lookup will gracefully handle the missing data.
        """
        max_last_time = None
        
        for stock_id in stock_ids:
            if stock_id not in quotes_dict or not quotes_dict[stock_id]:
                continue
            
            last_quote_time = self._parse_timestamp(quotes_dict[stock_id][-1]['timestamp'])
            
            if max_last_time is None or last_quote_time > max_last_time:
                max_last_time = last_quote_time
        
        return max_last_time
    
    def _process_time_segment(
        self,
        quotes_dict: Dict[int, List[Dict]],
        stock_ids: List[int],
        start_time: datetime,
        end_time: datetime,
        segment_id: int,
        total_segments: int,
        timestamps_dict: Dict[int, List[datetime]] = None,
        initial_prices: Dict[int, float] = None
    ) -> List[str]:
        """Process a time segment sequentially, advancing by minute intervals.
        
        Args:
            quotes_dict: Dictionary of stock_id -> sorted list of quotes
            stock_ids: List of stock IDs in order
            start_time: Starting timestamp for this segment
            end_time: Ending timestamp for this segment (barrier)
            segment_id: ID of this segment (for logging)
            total_segments: Total number of segments (for logging)
            timestamps_dict: Pre-parsed timestamps for each stock (for performance)
            initial_prices: Starting prices for this segment (to avoid boundary data loss)
        
        Returns:
            List of words generated in this segment
        """
        words = []
        current_time = start_time
        # Use initial_prices if provided (fixes segment boundary data loss)
        last_prices = initial_prices.copy() if initial_prices else {}
        minute_step = timedelta(minutes=1)
        
        logger.info(f"Segment {segment_id}/{total_segments}: Starting - {current_time}")
        
        skipped_minutes = 0
        while current_time <= end_time:
            # Skip non-market hours (weekends, holidays, before/after hours)
            if not self._is_market_open(current_time):
                # Jump to next market open instead of advancing minute-by-minute
                next_open = self._next_market_open(current_time)
                if next_open > end_time:
                    break
                skipped_minutes += int((next_open - current_time).total_seconds() / 60)
                current_time = next_open
                # Reset last_prices when jumping to new trading day
                # to avoid calculating deltas across non-trading periods
                last_prices = {}
                continue
            
            # Get quotes for all stocks at this time
            current_quotes = {}
            all_valid = True
            
            for stock_id in stock_ids:
                if stock_id not in quotes_dict:
                    all_valid = False
                    break
                
                # Use pre-parsed timestamps if available
                timestamps = timestamps_dict.get(stock_id) if timestamps_dict else None
                quote = self._get_quote_at_or_before(quotes_dict[stock_id], current_time, timestamps)
                if quote is None:
                    all_valid = False
                    break
                
                current_quotes[stock_id] = quote
            
            if all_valid:
                # Parse prices
                prices = {}
                for stock_id in stock_ids:
                    price = self.parse_price(current_quotes[stock_id]['price'])
                    if price is None:
                        all_valid = False
                        break
                    prices[stock_id] = price
                
                if all_valid:
                    # If we have previous prices, calculate deltas
                    if last_prices:
                        word = ""
                        for stock_id in stock_ids:
                            delta = self.calculate_delta(last_prices[stock_id], prices[stock_id])
                            symbol = self.delta_to_symbol(delta)
                            if symbol is None:
                                all_valid = False
                                break
                            word += symbol
                        
                        if all_valid:
                            words.append(word)
                    
                    # Update last prices for next iteration
                    last_prices = prices.copy()
            
            # Advance time by 1 minute
            current_time += minute_step
            
            # Check if we've moved past market close - jump to next open
            if not self._is_market_open(current_time) and current_time <= end_time:
                next_open = self._next_market_open(current_time)
                if next_open <= end_time:
                    skipped_minutes += int((next_open - current_time).total_seconds() / 60)
                    current_time = next_open
                    # Reset last_prices for new trading day
                    last_prices = {}
            
            # Progress logging every 1000 minutes
            if len(words) > 0 and len(words) % 1000 == 0:
                logger.debug(f"Segment {segment_id}/{total_segments}: Generated {len(words)} words")
        
        if skipped_minutes > 0:
            logger.info(f"Segment {segment_id}/{total_segments}: Skipped {skipped_minutes} non-market minutes")
        logger.info(f"Segment {segment_id}/{total_segments}: Completed with {len(words)} words")
        return words
    
    def extract_words_parallel(
        self,
        quotes_dict: Dict[int, List[Dict]],
        stock_ids: List[int]
    ) -> List[str]:
        """Extract words using parallel processing across time segments.
        
        Divides the time range into segments and processes each with a separate thread.
        Threads coordinate to avoid overlap and ensure consistency.
        Pre-parses timestamps for better binary search performance.
        Passes initial prices to each segment to avoid boundary data loss.
        """
        if not quotes_dict or not stock_ids:
            return []
        
        # Sort quotes and pre-parse timestamps for binary search performance
        timestamps_dict = {}
        for stock_id in quotes_dict:
            quotes_dict[stock_id].sort(
                key=lambda q: self._parse_timestamp(q['timestamp'])
            )
            # Pre-parse timestamps once (avoids repeated parsing in binary search)
            timestamps_dict[stock_id] = [
                self._parse_timestamp(q['timestamp']) for q in quotes_dict[stock_id]
            ]
        
        # Find time boundaries
        start_time = self._find_start_time(quotes_dict, stock_ids)
        end_time = self._find_end_time(quotes_dict, stock_ids)
        
        if start_time is None or end_time is None:
            logger.warning("Could not determine time boundaries")
            return []
        
        logger.info(f"Time range: {start_time} to {end_time}")
        logger.info(f"Duration: {(end_time - start_time).total_seconds() / 3600:.1f} hours")
        
        # Calculate segment boundaries
        total_duration = end_time - start_time
        num_segments = min(self.num_threads, 4)  # Use up to 4 segments
        segment_duration = total_duration / num_segments
        
        # Define segment start/end times
        segments = []
        for i in range(num_segments):
            seg_start = start_time + (segment_duration * i)
            if i == num_segments - 1:
                seg_end = end_time
            else:
                seg_end = start_time + (segment_duration * (i + 1))
            segments.append((seg_start, seg_end, i + 1))
        
        # Pre-compute initial prices for each segment to avoid boundary data loss
        segment_initial_prices = {}
        for seg_start, seg_end, seg_id in segments:
            if seg_id == 1:
                # First segment has no initial prices
                segment_initial_prices[seg_id] = None
            else:
                # Get prices at the minute before this segment starts
                initial_time = seg_start - timedelta(minutes=1)
                initial_prices = {}
                all_valid = True
                
                for stock_id in stock_ids:
                    if stock_id not in quotes_dict:
                        all_valid = False
                        break
                    timestamps = timestamps_dict.get(stock_id)
                    quote = self._get_quote_at_or_before(quotes_dict[stock_id], initial_time, timestamps)
                    if quote is None:
                        all_valid = False
                        break
                    price = self.parse_price(quote['price'])
                    if price is None:
                        all_valid = False
                        break
                    initial_prices[stock_id] = price
                
                segment_initial_prices[seg_id] = initial_prices if all_valid else None
        
        logger.info(f"Processing {num_segments} segments in parallel")
        
        # Process segments in parallel
        all_words = []
        with ThreadPoolExecutor(max_workers=num_segments) as executor:
            futures = {}
            for seg_start, seg_end, seg_id in segments:
                future = executor.submit(
                    self._process_time_segment,
                    quotes_dict,
                    stock_ids,
                    seg_start,
                    seg_end,
                    seg_id,
                    num_segments,
                    timestamps_dict,
                    segment_initial_prices[seg_id]
                )
                futures[future] = seg_id
            
            # Collect results in order
            segment_words = {}
            for future in as_completed(futures):
                seg_id = futures[future]
                words = future.result()
                segment_words[seg_id] = words
        
        # Combine results in order
        for seg_id in range(1, num_segments + 1):
            all_words.extend(segment_words[seg_id])
        
        logger.info(f"Total words generated: {len(all_words)}")
        return all_words
    
    def extract_words(
        self,
        quotes_dict: Dict[int, List[Dict]],
        stock_ids: List[int]
    ) -> List[str]:
        """Extract all words from quote data using optimized parallel processing."""
        return self.extract_words_parallel(quotes_dict, stock_ids)
    
    def count_unique_words(
        self,
        words: List[str]
    ) -> Tuple[int, Set[str]]:
        """Count unique words and return the set."""
        unique_words = set(words)
        return len(unique_words), unique_words
