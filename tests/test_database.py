"""Unit tests for database.py module."""
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from database import StockDatabase


@pytest.fixture
def db_config():
    """Database configuration fixture."""
    return {
        'host': 'localhost',
        'port': 3306,
        'user': 'test_user',
        'password': 'test_pass',
        'database': 'test_db'
    }


class TestStockDatabase:
    """Test StockDatabase class."""
    
    @patch('database.mysql.connector.connect')
    def test_database_connection(self, mock_connect, db_config):
        """Test database connection."""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        
        db = StockDatabase(**db_config)
        db.connect()
        
        # Verify connect was called with correct parameters
        mock_connect.assert_called_once_with(**db_config)
        assert db.connection is not None
    
    def test_database_initialization(self, db_config):
        """Test database object initializes with config."""
        db = StockDatabase(**db_config)
        
        assert db.host == 'localhost'
        assert db.port == 3306
        assert db.user == 'test_user'
        assert db.database == 'test_db'
    
    @patch('database.mysql.connector.connect')
    def test_get_all_stocks_returns_list(self, mock_connect, db_config):
        """Test get_all_stocks returns a list."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {'id': 1, 'ticker': 'AAPL', 'name': 'Apple'},
            {'id': 2, 'ticker': 'GOOGL', 'name': 'Google'}
        ]
        mock_connect.return_value = mock_connection
        
        db = StockDatabase(**db_config)
        db.connect()
        
        stocks = db.get_all_stocks()
        
        assert isinstance(stocks, list)
        assert len(stocks) == 2
    
    @patch('database.mysql.connector.connect')
    def test_get_random_stocks_returns_list(self, mock_connect, db_config):
        """Test get_random_stocks returns list of stocks."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {'id': 1, 'ticker': 'AAPL', 'name': 'Apple'},
        ]
        mock_connect.return_value = mock_connection
        
        db = StockDatabase(**db_config)
        db.connect()
        
        stocks = db.get_random_stocks(count=5, min_quotes=100000)
        
        assert isinstance(stocks, list)
    
    @patch('database.mysql.connector.connect')
    def test_get_quotes_for_stocks_returns_dict(self, mock_connect, db_config):
        """Test get_quotes_for_stocks returns dict."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        
        # Mock different return values for different queries
        mock_cursor.fetchall.side_effect = [
            [
                {'id': 1, 'stock': 1, 'price': '100.50', 'timestamp': '2024-01-01 10:00:00'},
                {'id': 2, 'stock': 1, 'price': '100.75', 'timestamp': '2024-01-01 10:15:00'},
            ]
        ]
        mock_connect.return_value = mock_connection
        
        db = StockDatabase(**db_config)
        db.connect()
        
        quotes = db.get_quotes_for_stocks([1])
        
        assert isinstance(quotes, dict)
        assert 1 in quotes
    
    @patch('database.mysql.connector.connect')
    def test_get_stocks_by_tickers_returns_matching_stocks(self, mock_connect, db_config):
        """Test get_stocks_by_tickers returns stocks matching the provided tickers."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {'id': 1, 'ticker': 'AAPL', 'name': 'Apple Inc.'},
            {'id': 2, 'ticker': 'GOOGL', 'name': 'Alphabet Inc.'}
        ]
        mock_connect.return_value = mock_connection
        
        db = StockDatabase(**db_config)
        db.connect()
        
        stocks = db.get_stocks_by_tickers(['AAPL', 'GOOGL'])
        
        assert isinstance(stocks, list)
        assert len(stocks) == 2
        assert stocks[0]['ticker'] == 'AAPL'
        assert stocks[1]['ticker'] == 'GOOGL'
    
    @patch('database.mysql.connector.connect')
    def test_get_stocks_by_tickers_returns_empty_for_empty_list(self, mock_connect, db_config):
        """Test get_stocks_by_tickers returns empty list when given empty list."""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        
        db = StockDatabase(**db_config)
        db.connect()
        
        stocks = db.get_stocks_by_tickers([])
        
        assert isinstance(stocks, list)
        assert len(stocks) == 0
    
    @patch('database.mysql.connector.connect')
    def test_get_stocks_by_tickers_partial_match(self, mock_connect, db_config):
        """Test get_stocks_by_tickers returns only stocks found in database."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        # Only AAPL is found, INVALID_TICKER is not in database
        mock_cursor.fetchall.return_value = [
            {'id': 1, 'ticker': 'AAPL', 'name': 'Apple Inc.'}
        ]
        mock_connect.return_value = mock_connection
        
        db = StockDatabase(**db_config)
        db.connect()
        
        stocks = db.get_stocks_by_tickers(['AAPL', 'INVALID_TICKER'])
        
        assert isinstance(stocks, list)
        assert len(stocks) == 1
        assert stocks[0]['ticker'] == 'AAPL'
    
    @patch('database.mysql.connector.connect')
    def test_get_stocks_by_tickers_none_found(self, mock_connect, db_config):
        """Test get_stocks_by_tickers returns empty list when no tickers found."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_connect.return_value = mock_connection
        
        db = StockDatabase(**db_config)
        db.connect()
        
        stocks = db.get_stocks_by_tickers(['INVALID1', 'INVALID2'])
        
        assert isinstance(stocks, list)
        assert len(stocks) == 0


class TestDatabaseWithoutConnection:
    """Test database methods that don't require connection."""
    
    def test_database_default_parameters(self):
        """Test database initializes with default parameters when password provided."""
        db = StockDatabase(password='test_password')
        
        assert db.host == 'localhost'
        assert db.port == 3306
        assert db.user == 'tinker'
        assert db.database == 'tinker'
        assert db.password == 'test_password'
    
    def test_database_requires_password(self):
        """Test that StockDatabase raises error when password not provided."""
        with pytest.raises(TypeError):
            StockDatabase()


class TestUpdateMissingStockNames:
    """Test update_missing_stock_names method."""
    
    @patch('database.mysql.connector.connect')
    @patch('database.yf.Ticker')
    def test_update_missing_stock_names_fetches_from_yfinance(self, mock_ticker, mock_connect):
        """Test that missing names are fetched from yFinance."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Stock has no name
        mock_cursor.fetchone.return_value = {'id': 1, 'ticker': 'AAPL', 'name': None}
        
        # Mock yFinance response
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'longName': 'Apple Inc.'}
        mock_ticker.return_value = mock_ticker_instance
        
        db = StockDatabase(password='test')
        db.connect()
        db.update_missing_stock_names([1])
        
        # Should have called yFinance
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('database.mysql.connector.connect')
    @patch('database.yf.Ticker')
    def test_update_missing_stock_names_skips_existing(self, mock_ticker, mock_connect):
        """Test that stocks with existing names are skipped."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Stock already has name
        mock_cursor.fetchone.return_value = {'id': 1, 'ticker': 'AAPL', 'name': 'Apple Inc.'}
        
        db = StockDatabase(password='test')
        db.connect()
        db.update_missing_stock_names([1])
        
        # Should NOT have called yFinance
        mock_ticker.assert_not_called()
    
    @patch('database.mysql.connector.connect')
    def test_update_missing_stock_names_handles_not_found(self, mock_connect):
        """Test handling of stock ID not found in database."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Stock not found
        mock_cursor.fetchone.return_value = None
        
        db = StockDatabase(password='test')
        db.connect()
        
        # Should not raise error
        db.update_missing_stock_names([999])
