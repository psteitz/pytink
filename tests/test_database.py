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
