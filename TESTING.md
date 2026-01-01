# Testing Guide for PyTink Stock Prediction System

## Overview

The PyTink project includes a comprehensive test suite with **68 unit and integration tests** covering all major components:

- **Processor Tests** (27 tests): Delta encoding, quantization, price calculations, binary search, segment boundaries
- **Model Tests** (22 tests): Dataset, collate function, transformer model, tensor pre-conversion
- **Database Tests** (7 tests): Database connection and queries
- **Integration Tests** (13 tests): End-to-end workflows, config management, artifacts

## Running Tests

### Run all tests:
```bash
cd ~/pytink
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_processor.py -v
pytest tests/test_model.py -v
pytest tests/test_database.py -v
pytest tests/test_integration.py -v
```

### Run specific test class:
```bash
pytest tests/test_processor.py::TestPriceProcessor -v
pytest tests/test_model.py::TestStockWordDataset -v
```

### Run specific test method:
```bash
pytest tests/test_processor.py::TestPriceProcessor::test_calculate_delta_positive_change -v
```

## Test Structure

```
tests/
├── __init__.py                 # Package initialization
├── test_processor.py           # Tests for processor.py (27 tests)
├── test_model.py               # Tests for model.py (22 tests)
├── test_database.py            # Tests for database.py (7 tests)
└── test_integration.py         # Integration tests (13 tests)
```

## Test Categories

### 1. Processor Tests (test_processor.py)

**Delta Value Constants (3 tests)**
- `test_delta_values_are_correct`: Verify DELTA_VALUES matches spec [-.01, -.005, -.001, 0, .001, .005, .01]
- `test_delta_to_char_mapping_size`: Ensure all 7 deltas map to characters
- `test_delta_to_char_uses_lowercase_letters`: Verify mapping uses a-g

**PriceProcessor Methods (8 tests)**
- `test_processor_initialization`: Verify processor initializes with correct interval
- `test_calculate_delta_positive_change`: Test delta calculation for price increase (100→101)
- `test_calculate_delta_negative_change`: Test delta calculation for price decrease (101→100)
- `test_calculate_delta_no_change`: Test delta calculation when prices equal (100→100)
- `test_calculate_delta_zero_old_price`: Test delta returns None for zero old price
- `test_symbol_to_delta_all_symbols`: Test all symbols (a-g) convert to deltas
- `test_symbol_to_delta_roundtrip`: Test roundtrip symbol→delta conversion
- `test_delta_to_symbol_nearest_neighbor`: Test quantization with 0.007→f (0.005)

**Quantization Boundaries (7 tests)**
- `test_delta_to_symbol_exact_match`: Test exact delta matches (e.g., -0.01→a)
- `test_delta_to_symbol_none_input`: Test None input returns None
- `test_delta_to_symbol_clamping_low`: Test extreme low deltas (-0.05) clamp to 'a'
- `test_delta_to_symbol_clamping_high`: Test extreme high deltas (0.05) clamp to 'g'
- `test_count_unique_words`: Test counting unique word sequences
- `test_count_unique_words_empty_list`: Test empty word list handling
- `test_quantization_boundaries`: Test midpoint quantization (6 boundary cases)

### 2. Model Tests (test_model.py)

**StockWordDataset (7 tests)**
- `test_dataset_initialization`: Verify dataset initializes correctly
- `test_dataset_length`: Verify len(dataset) = len(words) - sequence_length
- `test_dataset_getitem_returns_tuple`: Verify __getitem__ returns (input_ids, label) tuple
- `test_dataset_getitem_shapes`: Verify shapes are (seq_length,) and ()
- `test_dataset_getitem_values`: Verify token values are valid vocab IDs
- `test_dataset_empty_words`: Test dataset with empty word list
- `test_dataset_insufficient_words`: Test dataset when words < sequence_length

**custom_collate_fn (2 tests)**
- `test_collate_fn_batch_stacking`: Verify batch stacking produces (batch_size, seq_length) shape
- `test_collate_fn_tensor_types`: Verify output types are torch.LongTensor

**StockTransformerModel (7 tests)**
- `test_model_initialization`: Verify model initializes correctly
- `test_model_forward_pass`: Test forward pass without labels
- `test_model_forward_with_labels`: Test forward pass with labels (computes loss)
- `test_model_logits_shape`: Verify logits shape is (batch_size, seq_length, vocab_size)
- `test_model_predict`: Test prediction function output shape
- `test_model_train_eval_modes`: Test switching between train/eval modes
- `test_model_parameters`: Verify model has trainable parameters
- `test_model_device_placement`: Verify model is on correct device

### 3. Database Tests (test_database.py)

**Database Connection & Initialization (3 tests)**
- `test_database_initialization`: Verify database initializes with config
- `test_database_default_parameters`: Verify default config uses localhost:3306
- `test_database_connection`: Mock test for connection with credentials

**Database Query Methods (3 tests)**
- `test_get_all_stocks_returns_list`: Verify get_all_stocks() returns list
- `test_get_random_stocks_returns_list`: Verify get_random_stocks(count, min_quotes) returns list
- `test_get_quotes_for_stocks_returns_dict`: Verify get_quotes_for_stocks() returns dict with stock IDs as keys

### 4. Integration Tests (test_integration.py)

**Processor-to-Dataset Workflow (6 tests)**
- `test_processor_generates_valid_words`: Verify processor generates valid word sequences
- `test_delta_calculation_sequence`: Verify delta calculations form correct sequence
- `test_word_to_delta_roundtrip`: Test delta→symbol→delta roundtrip conversion
- `test_dataset_creation_from_words`: Test creating dataset from processor output
- `test_dataset_produces_valid_tensors`: Verify dataset produces valid input/label tensors
- `test_multiple_stocks_dataset`: Test dataset creation with multiple stocks
- `test_vocab_size_matches_delta_levels`: Verify vocabulary size matches delta levels

**Configuration Management (2 tests)**
- `test_yaml_config_creation`: Test creating and reading YAML config files
- `test_config_merging`: Test merging YAML config with CLI overrides

**Artifact Handling (3 tests)**
- `test_save_vocabulary`: Test saving vocabulary to JSON
- `test_save_predictions`: Test saving predictions to JSON
- `test_save_training_history`: Test saving training history to JSON

## Test Execution Output

```
======================== 68 passed in 3.75s ========================
```

All tests pass with no failures or errors.

## Coverage Analysis

### Processor Coverage
- ✓ Delta value constants and mappings
- ✓ Price delta calculation (positive, negative, zero changes)
- ✓ Delta to symbol quantization (exact, nearest-neighbor, clamping)
- ✓ Symbol to delta conversion and roundtrips
- ✓ Word generation and extraction
- ✓ Boundary conditions and edge cases

### Model Coverage
- ✓ Dataset initialization and length calculation
- ✓ Dataset indexing and tensor shape validation
- ✓ Token ID validation (within vocab range)
- ✓ Batch collation and stacking
- ✓ Model forward pass (with and without labels)
- ✓ Loss computation
- ✓ Model parameter count and device placement
- ✓ Train/eval mode switching

### Database Coverage
- ✓ Connection initialization
- ✓ Stock query methods
- ✓ Quote retrieval and filtering

### Integration Coverage
- ✓ End-to-end processor→dataset workflow
- ✓ Configuration file handling
- ✓ Multi-stock dataset creation
- ✓ Artifact persistence (model, vocab, predictions)

## Key Testing Patterns

### 1. Mocking External Dependencies
Database tests use `unittest.mock` to mock MySQL connections without requiring a live database:

```python
@patch('database.mysql.connector.connect')
def test_database_connection(self, mock_connect):
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection
    # Test code here
```

### 2. Parameterized Tests with subTest
Quantization tests use `subTest` to test multiple cases:

```python
test_cases = [
    (-0.0075, 'b'),
    (-0.003, 'b'),
    # ... more cases
]
for delta, expected_symbol in test_cases:
    with self.subTest(delta=delta):
        symbol = self.processor.delta_to_symbol(delta)
        self.assertEqual(symbol, expected_symbol)
```

### 3. Fixture Setup and Teardown
Integration tests use `setUp()` and `tearDown()` for resource management:

```python
def setUp(self):
    """Create test processor and mock data."""
    self.processor = PriceProcessor()
    self.mock_stocks = {1: [...]}

def tearDown(self):
    """Clean up temporary files."""
    self.temp_dir.cleanup()
```

### 4. Conditional Testing
Tests handle cases where data generation may vary:

```python
words = self.processor.extract_words_parallel({1: quotes}, [1])
if len(words) > 0:
    # Test only if words were generated
    dataset = StockWordDataset(words=words, vocab=vocab, sequence_length=4)
```

## Continuous Integration

To add tests to a CI/CD pipeline (e.g., GitHub Actions):

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python -m unittest discover tests/ -v
```

## Debugging Failed Tests

### 1. Run with more verbosity:
```bash
pytest tests/test_processor.py::TestPriceProcessor::test_calculate_delta_positive_change -v
```

### 2. Print intermediate values:
```python
def test_something(processor):
    result = processor.calculate_delta(100, 101)
    print(f"Result: {result}")  # Prints to console with -s flag
    assert result == 0.01
```

### 3. Use pytest with debugging:
```bash
pytest tests/test_processor.py::TestPriceProcessor::test_calculate_delta_positive_change -v -s --pdb
```

## Test Maintenance

### Adding New Tests
1. Create test method in appropriate test file
2. Use descriptive names: `test_<function>_<scenario>`
3. Add docstring explaining what is tested
4. Include setup/teardown if needed
5. Run tests to verify they pass

### Updating Tests
When code changes break tests:
1. Review the code change
2. Update test expectations if behavior changed intentionally
3. Update test if test logic was flawed
4. Re-run all tests to ensure no regressions

### Test File Organization
- Group related tests in classes
- Keep test files aligned with source files
- Use descriptive test method names
- Document non-obvious test logic

## Performance Notes

All 53 tests complete in ~1.0 second on a standard system. Tests are designed to be:
- **Fast**: Use small datasets and mock external dependencies
- **Isolated**: Each test is independent
- **Repeatable**: Deterministic results

## Next Steps

Potential test enhancements:
1. Add performance benchmarking tests
2. Add stress tests with large datasets
3. Add property-based tests using hypothesis
4. Add end-to-end tests with real database (marked as slow)
5. Generate test coverage reports

## References

- [Python unittest documentation](https://docs.python.org/3/library/unittest.html)
- [Mock and patch objects](https://docs.python.org/3/library/unittest.mock.html)
- Project source code:
  - `src/processor.py`: Delta encoding and quantization
  - `src/model.py`: PyTorch dataset and transformer model
  - `src/database.py`: MySQL database interface
  - `train_model.py`: Main CLI and config handling
