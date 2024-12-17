import torch
from models.bart_model import BARTModel
from models.config import DEVICE, MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG

def test_config():
    print("Testing config.py...")

    # Print DEVICE
    print(f"DEVICE: {DEVICE}")

    # Print Model Configurations
    print("\nMODEL CONFIGURATION:")
    for key, value in MODEL_CONFIG.items():
        print(f"{key}: {value}")

    # Print Training Configurations
    print("\nTRAINING CONFIGURATION:")
    for key, value in TRAINING_CONFIG.items():
        print(f"{key}: {value}")

    # Print Data Configurations
    print("\nDATA CONFIGURATION:")
    for key, value in DATA_CONFIG.items():
        print(f"{key}: {value}")

def test_bart_model():
    # Set device
    device = DEVICE
    print(f"\nUsing device: {device}")

    # Model parameters from config
    vocab_size = MODEL_CONFIG['vocab_size']
    d_model = MODEL_CONFIG['d_model']
    nhead = MODEL_CONFIG['nhead']
    num_encoder_layers = MODEL_CONFIG['num_encoder_layers']
    num_decoder_layers = MODEL_CONFIG['num_decoder_layers']

    # Create dummy input data
    batch_size = 2
    seq_len = 10
    src = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # Initialize the BART model
    print("\nInitializing BART Model...")
    model = BARTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    ).to(device)

    print("Model initialized successfully!")

    # Forward pass
    print("\nRunning a forward pass...")
    logits = model(src, tgt)
    print(f"Output logits shape: {logits.shape}")  # Expect shape [batch_size, seq_len, vocab_size]

if __name__ == "__main__":
    # Test config.py
    test_config()

    # Test BART model
    test_bart_model()
