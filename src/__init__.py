"""Stock Price Prediction with Transformers

A sequence-to-sequence transformer model for predicting stock price movements.
"""

__version__ = "0.1.0"

from .database import StockDatabase
from .processor import PriceProcessor, DELTA_VALUES, DELTA_TO_CHAR, CHAR_TO_DELTA
from .model import StockWordDataset, StockTransformerModel
from .analysis import (
    plot_training_loss,
    plot_epoch_loss,
    plot_word_frequency,
    analyze_prediction_quality,
    save_vocabulary,
    load_vocabulary
)

__all__ = [
    'StockDatabase',
    'PriceProcessor',
    'DELTA_VALUES',
    'DELTA_TO_CHAR',
    'CHAR_TO_DELTA',
    'StockWordDataset',
    'StockTransformerModel',
    'plot_training_loss',
    'plot_epoch_loss',
    'plot_word_frequency',
    'analyze_prediction_quality',
    'save_vocabulary',
    'load_vocabulary',
]
