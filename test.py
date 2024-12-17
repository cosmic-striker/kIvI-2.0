from models.bart_model import BARTModel
from models.tokenizer import Tokenizer
from utils.helper import save_model, load_model, print_summary
from models.config import config

def test_model():
    """
    Test the BARTModel initialization and summary.
    """
    print("=== Testing Model Creation ===")
    vocab_size = config['vocab_size']  # Load vocab_size from config
    try:
        model = BARTModel(vocab_size=vocab_size)
        print("Model created successfully!")
        print_summary(model)
    except Exception as e:
        print(f"Error during model creation: {e}")

def test_tokenizer():
    """
    Test the Tokenizer training and encoding.
    """
    print("\n=== Testing Tokenizer ===")
    train_data_path = "data/raw/sample_train.txt"  # Update with your training data path
    vocab_size = config['vocab_size']
    try:
        tokenizer = Tokenizer(train_data_path=train_data_path, vocab_size=vocab_size)
        print("Tokenizer trained successfully!")
        test_text = "This is a test sentence."
        encoded_text = tokenizer.encode(test_text)
        decoded_text = tokenizer.decode(encoded_text)
        print(f"Test Sentence: {test_text}")
        print(f"Encoded: {encoded_text}")
        print(f"Decoded: {decoded_text}")
    except Exception as e:
        print(f"Error during tokenizer testing: {e}")

def test_save_load_model():
    """
    Test saving and loading the model.
    """
    print("\n=== Testing Save/Load Model ===")
    vocab_size = config['vocab_size']
    save_path = "checkpoints/test_model.txt"
    try:
        # Initialize model
        model = BARTModel(vocab_size=vocab_size)
        print("Model initialized successfully!")

        # Save model
        save_model(model, save_path)

        # Load model
        loaded_model = BARTModel(vocab_size=vocab_size)
        loaded_model = load_model(loaded_model, save_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error during save/load model test: {e}")

def main():
    """
    Run all tests.
    """
    print("=== Running Tests ===")
    test_model()
    test_tokenizer()
    test_save_load_model()

if __name__ == "__main__":
    main()
