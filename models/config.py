import torch

# General Configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU
SEED = 42  # Random seed for reproducibility

# Model Configuration
MODEL_CONFIG = {
    "vocab_size": 30522,      # Vocabulary size (adjust as per your tokenizer)
    "d_model": 768,           # Model dimension (hidden size)
    "nhead": 12,              # Number of attention heads
    "num_encoder_layers": 12, # Number of encoder layers
    "num_decoder_layers": 12, # Number of decoder layers
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 8,            # Batch size
    "num_epochs": 10,           # Number of training epochs
    "learning_rate": 5e-5,      # Learning rate
    "weight_decay": 0.01,       # Weight decay for regularization
    "gradient_clip": 1.0,       # Gradient clipping to avoid exploding gradients
    "save_checkpoint": True,    # Save checkpoints during training
    "checkpoint_path": "checkpoints/bart_checkpoint.pt",  # Path to save checkpoints
}

# Data Configuration
DATA_CONFIG = {
    "max_seq_length": 512,     # Maximum sequence length for tokenized data
    "train_data_path": "data/processed/train_data.pt",  # Processed training data
    "val_data_path": "data/processed/val_data.pt",      # Processed validation data
    "tokenizer_path": "models/tokenizer.json",          # Path to tokenizer (custom or prebuilt)
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_interval": 50,      # Log training loss every n batches
    "log_file": "results/training.log"  # File to store training logs
}

# Distributed Training Configuration
DISTRIBUTED_CONFIG = {
    "use_ddp": True,         # Enable Distributed Data Parallel (DDP) for multi-GPU training
    "world_size": 3,         # Total number of processes (e.g., 3 laptops)
    "backend": "nccl",       # Backend for communication (use "nccl" for GPUs)
    "master_ip": "127.0.0.1", # Master node's IP (adjust as per your setup)
    "master_port": "29500",  # Port for distributed communication
}

config = {
    "vocab_size": 1000,
    "hidden_size": 768,
    "num_layers": 6,
    "num_heads": 8,
    "ffn_dim": 2048,
    "dropout": 0.1
}

