# PYTINK - Stock Price Prediction with Transformers

## Welcome! 👋

This is a transformer model to predict stock price movements using sequences of changes in a list of stocks.
The alphabet encodes changes. The default model uses configuration in [config_template.yaml](config_template.yaml).

---

## 📋 START HERE

Choose your learning path:

### **New to the project?**
1. Follow **[QUICKSTART.md](QUICKSTART.md)** (setup & first run)
2. Try **[EXAMPLES.md](EXAMPLES.md)** (11 practical examples)

### **Want to dive in immediately?**
```bash
pip install -r requirements.txt
pip install -e .
python test_installation.py          # Verify setup
pytink-train --db-password YOUR_PASSWORD   # Run with defaults
```

### **Looking for specific information?**
- **Technical docs**: See [README.md](README.md)
- **Configuration**: See [config_template.yaml](config_template.yaml)
- **Source code**: See [src/](src/) directory

---

## 🎯 What This Project Does

Trains a **transformer model** to predict the next "word" in a sequence of stock price changes:

- **Input**: List of stocks (e.g., ["AAPL", "MSFT", "GOOGL"])
- **Process**: Encode price changes as letters (a-g for 7 delta levels)
- **Example Word**: "acgaeb" = 6 stocks with specific price movements
- **Model Task**: Given 4-16 words, predict the next word
- **Output**: Predictions of next period's price movements

---

## 📁 Project Structure

```
pytink/
├── README.md                 ← Full technical documentation
├── QUICKSTART.md             ← Setup & usage guide
├── EXAMPLES.md               ← 11 usage examples
├── STRUCTURE.txt             ← File descriptions
├── INDEX.md                  ← You are here
│
├── pyproject.toml            ← Package metadata & entry points
├── requirements.txt          ← Install with: pip install -r requirements.txt
├── config_template.yaml      ← Configuration template
├── test_installation.py      ← Verify installation
│
├── src/                      ← Core Python package
│   └── pytink/
│       ├── database.py       ← MySQL interface
│       ├── processor.py      ← Data processing & encoding
│       ├── model.py          ← PyTorch models
│       ├── analysis.py       ← Visualization tools
│       ├── farming.py        ← Automated model generation
│       ├── train_model.py    ← Training CLI (entry: pytink-train)
│       ├── inference.py      ← Evaluate trained models (entry: pytink-infer)
│       └── __init__.py       ← Package init
│
└── tests/                    ← Unit & integration tests
    ├── test_database.py
    ├── test_processor.py
    ├── test_model.py
    ├── test_integration.py
    ├── test_inference.py
    └── test_farming.py
```

---

## 🚀 Quick Start (2 minutes)

### Command Line (Recommended)
```bash
cd ~/pytink
pip install -r requirements.txt
pip install -e .
pytink-train --db-password YOUR_PASSWORD
```

### Evaluate a Trained Model
```bash
pytink-infer --db-password YOUR_PASSWORD --model-dir models/TICKER-LIST/TIMESTAMP/
```

### Verify Installation First
```bash
cd ~/pytink
python test_installation.py
```

### Run Tests
```bash
pytest tests/ -v
```

---

## 📚 Documentation Files

| File | Purpose | Read Time |
|------|---------|-----------|

| **QUICKSTART.md** | Setup, basic usage, troubleshooting | 10 min |
| **README.md** | Complete technical reference | 20 min |
| **EXAMPLES.md** | 11 practical code examples | 15 min |
| **STRUCTURE.txt** | File descriptions & statistics | 5 min |

---

## 🔧 Core Modules

### `pytink.database` - MySQL Interface
```python
from pytink.database import StockDatabase
db = StockDatabase()
db.connect()
stocks = db.get_random_stocks(count=10)
quotes = db.get_quotes_for_stocks(stock_ids)
```

### `pytink.processor` - Data Processing
```python
from pytink.processor import PriceProcessor
processor = PriceProcessor(interval_minutes=15)
words = processor.extract_words(quotes, stock_ids)
unique_count, unique_words = processor.count_unique_words(words)
```

### `pytink.model` - PyTorch Models
```python
from pytink.model import StockWordDataset, StockTransformerModel
dataset = StockWordDataset(words, vocab, context_window_size=4)
model = StockTransformerModel(vocab_size=len(vocab))
predictions = model.predict(input_ids)
```

### `pytink.analysis` - Visualization
```python
from pytink.analysis import plot_training_loss, analyze_prediction_quality
plot_training_loss(history)
analyze_prediction_quality(predictions)
```

---

## 📊 Delta Encoding (Key Concept)

Price changes are encoded as letters a-g:

| Letter | Delta | Change |
|--------|-------|--------|
| **a** | -0.01 | ↓ 1% or more |
| **b** | -0.005 | ↓ 0.5% |
| **c** | -0.001 | ↓ 0.1% |
| **d** | 0.00 | → 0% |
| **e** | +0.001 | ↑ 0.1% |
| **f** | +0.005 | ↑ 0.5% |
| **g** | +0.01 | ↑ 1% or more |

**Example**: "acgaeb" with 6 stocks = AAPL↓1%, AAL↓0.1%, MSFT↑1%, GOOGL↓1%, AMZN↑0.1%, TSLA↓0.5%

---

## 📈 Workflow

```
MySQL Database
    ↓ (fetch historical prices)
Raw Quote Data (timestamps + prices)
    ↓ (align timestamps, calculate deltas)
Encoded "Words" (sequences like "acgaeb")
    ↓ (create training pairs)
PyTorch Dataset
    ↓ (train transformer)
Trained Model
    ↓ (evaluate performance)
Loss / Accuracy / Perplexity metrics
```

---

## 💻 System Requirements

- **Python**: 3.8+
- **Database**: MySQL 5.7+ (local, port 3306)
- **RAM**: 4GB minimum (8GB+ recommended)
- **GPU**: Optional (training faster with CUDA)
- **Disk**: ~500MB for dependencies

---

## 🎓 Learning Path

### Beginner
1. Read PROJECT_SUMMARY.md
2. Run `python test_installation.py`
3. Run `python train_model.py`
4. Review EXAMPLES.md (Example 1-3)

### Intermediate
1. Review README.md
2. Run `pytest tests/`
3. Modify parameters in QUICKSTART.md
4. Try EXAMPLES.md (Example 4-7)

### Advanced
1. Study source code in src/
2. Implement custom modifications
3. Review EXAMPLES.md (Example 8-11)
4. Experiment with config_template.py

---

## 🤔 FAQ

**Q: What's a "word" in this context?**
A: A sequence of letters representing price changes for all stocks in one time period. Example: "acgaeb" is a 6-stock word.

**Q: How many "words" will there be?**
A: Depends on data. Usually 100-10,000 unique words. With 10 stocks, max theoretical is 9^10 (3.5B), but real data has far fewer.

**Q: Can I use different stocks?**
A: Yes! Use `db.get_random_stocks(count=X)` or query specific tickers in database.py.

**Q: How long does training take?**
A: ~1-5 minutes per epoch (10 epochs default), depending on data size and hardware.

**Q: How do I save the trained model?**
A: See EXAMPLES.md (Example 10) for model saving/loading code.

**Q: Can I use this for real trading?**
A: This is a research/educational project. Don't use for real trading without extensive testing.

**Q: What if I don't have MySQL?**
A: The database module handles connections. You'll need MySQL running on localhost:3306 with the "tinker" database.

---

## 🔗 Quick Links

- **Setup**: [QUICKSTART.md](QUICKSTART.md)
- **Examples**: [EXAMPLES.md](EXAMPLES.md)
- **Technical Docs**: [README.md](README.md)
- **Source Code**: [src/](src/)
- **Configuration**: [config_template.py](config_template.py)
- **Project Overview**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **File Structure**: [STRUCTURE.txt](STRUCTURE.txt)

---

## 🧪 Testing

Verify installation before starting:

```bash
python test_installation.py
```

This checks:
- ✓ Python packages
- ✓ Project modules
- ✓ Database connection
- ✓ PyTorch setup
- ✓ Model creation
- ✓ Data processing

---

## 📝 Files at a Glance

| File | Type | Purpose |
|------|------|---------|
| train_model.py | Script | CLI interface |
| test_installation.py | Script | Installation verification |
| config_template.py | Config | Configuration templates |
| requirements.txt | Config | Dependencies |
| src/database.py | Code | Database access |
| src/processor.py | Code | Data processing |
| src/model.py | Code | PyTorch models |
| src/analysis.py | Code | Visualization |
| tests/ | Tests | Unit & integration tests (pytest) |
| README.md | Docs | Full technical reference |
| QUICKSTART.md | Docs | Setup & usage guide |
| EXAMPLES.md | Docs | 11 code examples |
| PROJECT_SUMMARY.md | Docs | Project overview |
| STRUCTURE.txt | Docs | File descriptions |

---

## 🎯 Next Steps

1. **Now**: Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) (5 min)
2. **Then**: Follow [QUICKSTART.md](QUICKSTART.md) (10 min)
3. **Test**: Run `python test_installation.py` (1 min)
4. **Try**: Run `python train_model.py` (5-10 min)
5. **Explore**: Check [EXAMPLES.md](EXAMPLES.md) for more use cases

---

## 📞 Support

- **Installation issues**: See [QUICKSTART.md](QUICKSTART.md) Troubleshooting
- **Usage questions**: See [EXAMPLES.md](EXAMPLES.md)
- **Technical details**: See [README.md](README.md)
- **Configuration**: See [config_template.py](config_template.py)

---

## 📄 License

Educational project for stock price prediction research.

---

**Version**: 0.1.0  
**Created**: December 29, 2025  
**Status**: Complete & Ready to Use ✓

