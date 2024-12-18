import torch



config = {
    # Directory paths
    "checkpoint_dir": "checkpoints",  # Directory to save model checkpoints
    "logs_dir": "logs",  # Directory to save logs
    "data_dir": "data",  # Root directory for datasets
    "raw_data_dir": "data/raw",  # Directory for raw datasets
    "processed_data_dir": "data/processed",  # Directory for processed datasets
    "tokenizer_dir": "tokenizers",  # Directory for tokenizer files

    # Model training configurations
    "batch_size": 32,  # Training batch size
    "num_epochs": 10,  # Number of training epochs
    "learning_rate": 0.001,  # Learning rate for optimizer
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Device to use for training
    "seed": 42,  # Random seed for reproducibility

    # File paths
    "final_model_name": "final_trained_model.pth",  # Name of the final trained model file
    "tokenizer_file": "tokenizer.json",  # Default tokenizer file name
    "test_data_file": "test_data.txt",  # Default test data file name
}
