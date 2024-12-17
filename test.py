#import os
#import sys
from models.bart_model import BARTModel
from models.tokenizer import Tokenizer

from scripts.evaluate import evaluate  # Import evaluation function

def test_model():
    print("=== Testing Model Creation ===")
    model = BARTModel()
    print("Model created successfully!")

def test_tokenizer():
    print("=== Testing Tokenizer Training ===")
    train_data_path = "data/raw/sample_train.txt"
    tokenizer = Tokenizer(train_data_path=train_data_path, vocab_size=50)
    tokenizer.save("models/tokenizer.model")
    print("Tokenizer trained and saved successfully!")

def test_evaluate():
    print("=== Testing Model Evaluation ===")
    model_path = "results/checkpoints/best_model.pt"  # Path to saved model
    test_data_path = "data/raw/test.txt"              # Test data path
    tokenizer_path = "models/tokenizer.model"         # Tokenizer path

    # Run evaluation
    evaluate(model_path, test_data_path, tokenizer_path)

if __name__ == "__main__":
    print("Running tests...")

    # Run individual tests
    test_model()
    test_tokenizer()
    test_evaluate()

    print("All tests completed successfully!")
