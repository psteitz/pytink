"""
Configuration template for stock prediction experiments.
Copy and modify to customize your analysis.
"""

# Database Configuration
# Note: Password must be provided via --db-password command-line argument
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'tinker',
    'database': 'tinker'
}

# Data Processing Configuration
DATA_CONFIG = {
    'num_stocks': 10,              # Number of random stocks to select
    'interval_minutes': 15,        # Time interval between quotes
    'sequence_length': 4,          # Number of words as context
}

# Model Configuration
MODEL_CONFIG = {
    'hidden_size': 128,            # Transformer hidden dimension
    'num_hidden_layers': 4,        # Number of transformer blocks
    'num_attention_heads': 4,      # Number of attention heads
    'max_position_embeddings': 256, # Maximum sequence length
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,              # Training batch size
    'num_epochs': 10,              # Number of training epochs
    'learning_rate': 1e-4,         # Adam optimizer learning rate
    'weight_decay': 0,             # L2 regularization (optional)
    'warmup_steps': 100,           # Learning rate warmup (optional)
}

# Experiment Presets
EXPERIMENTS = {
    'quick_test': {
        'num_stocks': 5,
        'batch_size': 16,
        'num_epochs': 2,
    },
    'standard': {
        'num_stocks': 10,
        'batch_size': 32,
        'num_epochs': 10,
    },
    'large': {
        'num_stocks': 20,
        'batch_size': 64,
        'num_epochs': 20,
    },
    'deep_model': {
        'hidden_size': 256,
        'num_hidden_layers': 8,
        'num_attention_heads': 8,
    },
    'aggressive_learning': {
        'learning_rate': 5e-4,
        'batch_size': 16,
    },
}

# Delta Encoding (used by processor)
DELTA_RANGES = [
    -0.10,   # a: -10% or less
    -0.05,   # b: -5%
    -0.025,  # c: -2.5%
    -0.01,   # d: -1%
     0.00,   # e: 0%
     0.01,   # f: +1%
     0.025,  # g: +2.5%
     0.05,   # h: +5%
     0.10,   # i: +10% or more
]

# Output Configuration
OUTPUT_CONFIG = {
    'save_model': False,           # Save trained model
    'save_vocabulary': False,      # Save vocab to JSON
    'save_predictions': False,     # Save predictions to file
    'plot_results': False,         # Generate plots
    'verbose': True,               # Detailed logging
}
