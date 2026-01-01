#!/usr/bin/env python3
"""
Test script to verify the installation and basic functionality.
Run this after installing requirements to ensure everything works.
"""

import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    packages = {
        'mysql.connector': 'mysql-connector-python',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'torch': 'pytorch',
        'transformers': 'transformers',
    }
    
    success = True
    for module, package_name in packages.items():
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - MISSING")
            print(f"  Install with: pip install {package_name}")
            success = False
    
    return success

def test_module_imports():
    """Test that project modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Project Modules")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    modules = [
        ('database', 'StockDatabase'),
        ('processor', 'PriceProcessor'),
        ('model', 'StockWordDataset'),
        ('model', 'StockTransformerModel'),
        ('analysis', 'plot_training_loss'),
    ]
    
    success = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except (ImportError, AttributeError) as e:
            print(f"✗ {module_name}.{class_name}")
            success = False
    
    return success

def test_database_connection():
    """Test database connection (requires running MySQL)."""
    print("\n" + "=" * 60)
    print("Testing Database Connection")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    try:
        from database import StockDatabase
        db = StockDatabase()
        db.connect()
        print("✓ Database connection successful")
        
        stocks = db.get_all_stocks()
        print(f"✓ Found {len(stocks)} stocks in database")
        
        if len(stocks) > 0:
            print(f"✓ Sample stocks: {', '.join([s['ticker'] for s in stocks[:3]])}")
        
        db.close()
        print("✓ Database disconnected")
        return True
    
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("\nMake sure:")
        print("  - MySQL is running")
        print("  - Database 'tinker' exists")
        print("  - User 'tinker' has correct password")
        print("  - Port 3306 is accessible")
        return False

def test_pytorch_cuda():
    """Test PyTorch CUDA support."""
    print("\n" + "=" * 60)
    print("Testing PyTorch Configuration")
    print("=" * 60)
    
    import torch
    
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA is not available (will use CPU)")
    
    return True

def test_model_creation():
    """Test model creation."""
    print("\n" + "=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    try:
        from model import StockTransformerModel
        
        # Test small model
        model = StockTransformerModel(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2
        )
        
        print("✓ Model created successfully")
        
        params = sum(p.numel() for p in model.get_model().parameters())
        print(f"✓ Model has {params:,} parameters")
        
        return True
    
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_data_processing():
    """Test data processing pipeline."""
    print("\n" + "=" * 60)
    print("Testing Data Processing")
    print("=" * 60)
    
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    try:
        from processor import (
            PriceProcessor,
            DELTA_VALUES,
            DELTA_TO_CHAR,
            CHAR_TO_DELTA
        )
        
        processor = PriceProcessor(interval_minutes=15)
        print("✓ PriceProcessor initialized")
        
        # Test delta encoding
        test_deltas = [0.032, -0.015, 0.0, 0.08, -0.03]
        for delta in test_deltas:
            symbol = processor.delta_to_symbol(delta)
            back_to_delta = processor.symbol_to_delta(symbol)
            print(f"  {delta:.3f} → '{symbol}' → {back_to_delta:.3f}")
        
        print("✓ Delta encoding working")
        
        # Test price parsing
        test_prices = ['100.50', '99.99', '101.01']
        for price_str in test_prices:
            price = processor.parse_price(price_str)
            print(f"  '{price_str}' → {price:.2f}")
        
        print("✓ Price parsing working")
        
        return True
    
    except Exception as e:
        print(f"✗ Data processing test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PYTINK PROJECT - INSTALLATION TEST")
    print("=" * 60 + "\n")
    
    results = {
        'Imports': test_imports(),
        'Project Modules': test_module_imports(),
        'PyTorch': test_pytorch_cuda(),
        'Model Creation': test_model_creation(),
        'Data Processing': test_data_processing(),
    }
    
    # Database test is optional
    try:
        results['Database Connection'] = test_database_connection()
    except Exception as e:
        print(f"\n⚠ Database test skipped (MySQL may not be running)")
        results['Database Connection'] = None
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(v for v in results.values() if v is not None)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready to use!")
    else:
        print("✗ SOME TESTS FAILED - See details above")
        print("\nNext steps:")
        print("1. Check error messages above")
        print("2. Ensure MySQL is running for database tests")
        print("3. Run: pip install -r requirements.txt")
        print("4. See QUICKSTART.md for setup instructions")
    
    print("=" * 60 + "\n")
    
    return all_passed

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
