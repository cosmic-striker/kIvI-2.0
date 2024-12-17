import torch
from models.bart_model import BARTModel

def test_bart_model():
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define model parameters
    vocab_size = 30522  # Example vocab size (adjust as per your tokenizer)
    batch_size = 2
    seq_len = 10  # Example sequence length
    d_model = 768  # Dimension of the model (standard BART value)
    
    # Create dummy data (input and target sequences)
    src = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)  # Random source sequence
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)  # Random target sequence
    src_mask = None  # For simplicity, no masking in this test
    tgt_mask = None  # Same for target mask
    src_key_padding_mask = None  # No padding in this test
    tgt_key_padding_mask = None  # Same for target padding mask
    
    # Instantiate the BART model
    model = BARTModel(vocab_size=vocab_size, d_model=d_model).to(device)
    
    # Perform a forward pass through the model
    logits = model(src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
    
    # Print the shape of the output logits (should be [batch_size, tgt_len, vocab_size])
    print(f"Logits shape: {logits.shape}")
    
    # Optionally, print the logits themselves to inspect
    print("Logits (output of the model):")
    print(logits)

if __name__ == "__main__":
    test_bart_model()
