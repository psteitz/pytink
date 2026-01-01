"""Database utilities for stock data retrieval."""
import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Tuple
from datetime import datetime
import logging
import yfinance as yf

logger = logging.getLogger(__name__)


class StockDatabase:
    """Interface for accessing stock data from MySQL database."""
    
    def __init__(
        self,
        password: str,
        host: str = "localhost",
        port: int = 3306,
        user: str = "tinker",
        database: str = "tinker"
    ):
        """Initialize database connection parameters.
        
        Args:
            password: Database password (required)
            host: Database host (default: localhost)
            port: Database port (default: 3306)
            user: Database user (default: tinker)
            database: Database name (default: tinker)
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
    
    def connect(self):
        """Establish connection to the database."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            logger.info("Successfully connected to database")
        except Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")
    
    def get_all_stocks(self) -> List[Dict]:
        """Retrieve all stocks from the database."""
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute("SELECT id, ticker, name FROM stocks")
            stocks = cursor.fetchall()
            return stocks
        finally:
            cursor.close()
    
    def get_random_stocks(self, count: int = 10, min_quotes: int = 0) -> List[Dict]:
        """Retrieve a random sample of stocks with at least min_quotes quotes.
        
        Args:
            count: Number of stocks to return
            min_quotes: Minimum number of quotes required for a stock (default: 0, no minimum)
        
        Returns:
            List of stocks meeting the criteria
        """
        cursor = self.connection.cursor(dictionary=True)
        try:
            if min_quotes > 0:
                # Join with quotes table to filter by quote count
                query = f"""
                    SELECT s.id, s.ticker, s.name
                    FROM stocks s
                    WHERE (
                        SELECT COUNT(*) FROM quotes q WHERE q.stock = s.id
                    ) >= %s
                    ORDER BY RAND()
                    LIMIT %s
                """
                cursor.execute(query, (min_quotes, count))
            else:
                cursor.execute(f"SELECT id, ticker, name FROM stocks ORDER BY RAND() LIMIT {count}")
            
            stocks = cursor.fetchall()
            return stocks
        finally:
            cursor.close()
    
    def get_quotes_for_stock(
        self,
        stock_id: int,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Dict]:
        """Retrieve quotes for a specific stock, optionally filtered by date range."""
        cursor = self.connection.cursor(dictionary=True)
        try:
            if start_date and end_date:
                query = """
                    SELECT price, timestamp, stock 
                    FROM quotes 
                    WHERE stock = %s AND timestamp BETWEEN %s AND %s
                    ORDER BY timestamp ASC
                """
                cursor.execute(query, (stock_id, start_date, end_date))
            else:
                query = """
                    SELECT price, timestamp, stock 
                    FROM quotes 
                    WHERE stock = %s
                    ORDER BY timestamp ASC
                """
                cursor.execute(query, (stock_id,))
            
            quotes = cursor.fetchall()
            return quotes
        finally:
            cursor.close()
    
    def get_quotes_for_stocks(
        self,
        stock_ids: List[int],
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[int, List[Dict]]:
        """Retrieve quotes for multiple stocks using a single batched query.
        
        This is more efficient than making N separate queries.
        """
        if not stock_ids:
            return {}
        
        cursor = self.connection.cursor(dictionary=True)
        try:
            # Build parameterized query for all stocks at once
            placeholders = ','.join(['%s'] * len(stock_ids))
            
            if start_date and end_date:
                query = f"""
                    SELECT price, timestamp, stock 
                    FROM quotes 
                    WHERE stock IN ({placeholders}) AND timestamp BETWEEN %s AND %s
                    ORDER BY stock, timestamp ASC
                """
                params = tuple(stock_ids) + (start_date, end_date)
            else:
                query = f"""
                    SELECT price, timestamp, stock 
                    FROM quotes 
                    WHERE stock IN ({placeholders})
                    ORDER BY stock, timestamp ASC
                """
                params = tuple(stock_ids)
            
            cursor.execute(query, params)
            all_quotes = cursor.fetchall()
            
            # Group quotes by stock_id
            quotes_by_stock = {stock_id: [] for stock_id in stock_ids}
            for quote in all_quotes:
                stock_id = quote['stock']
                if stock_id in quotes_by_stock:
                    quotes_by_stock[stock_id].append(quote)
            
            return quotes_by_stock
        finally:
            cursor.close()

    def update_missing_stock_names(self, stock_ids: List[int]):
        """Fetch missing stock names from yFinance and update the database.
        
        Args:
            stock_ids: List of stock IDs to check and update
        """
        cursor = self.connection.cursor(dictionary=True)
        try:
            for stock_id in stock_ids:
                # Get the stock ticker
                query = "SELECT id, ticker, name FROM stocks WHERE id = %s"
                cursor.execute(query, (stock_id,))
                stock = cursor.fetchone()
                
                if not stock:
                    logger.warning(f"Stock ID {stock_id} not found in database")
                    continue
                
                # Skip if name already exists
                if stock['name'] is not None and stock['name'].strip():
                    continue
                
                # Fetch name from yFinance
                ticker = stock['ticker']
                try:
                    ticker_obj = yf.Ticker(ticker)
                    name = ticker_obj.info.get('longName') or ticker_obj.info.get('shortName') or ticker
                    
                    if name and name != ticker:
                        # Update the database
                        update_query = "UPDATE stocks SET name = %s WHERE id = %s"
                        cursor.execute(update_query, (name, stock_id))
                        self.connection.commit()
                        logger.info(f"Updated stock {ticker} with name: {name}")
                    else:
                        logger.warning(f"Could not fetch name for ticker {ticker} from yFinance")
                
                except Exception as e:
                    logger.warning(f"Error fetching data for ticker {ticker}: {e}")
        
        finally:
            cursor.close()
