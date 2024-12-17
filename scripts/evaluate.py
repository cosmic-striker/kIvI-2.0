import torch
from torch.utils.data import DataLoader, Dataset
from models.bart_model import BARTModel
from models.tokenizer import Tokenizer
from utils.helper import load_model  # Helper to load saved model
import os
from rouge_score import rouge_scorer  # For ROUGE metrics
from nltk.translate.bleu_score import sentence_bleu  # For BLEU score

# Custom Dataset for Evaluation
class TestDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.samples = self.load_data(data_path)

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_text = self.samples[idx]
        input_ids = self.tokenizer.encode(input_text)
        return torch.tensor(input_ids)

# Evaluation Script
def evaluate(model_path, test_data_path, tokenizer_path, batch_size=8):
    print("Starting evaluation...")

    # Load tokenizer
    tokenizer = Tokenizer(tokenizer_path=tokenizer_path)
    print("Tokenizer loaded.")

    # Load dataset
    dataset = TestDataset(test_data_path, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BARTModel()
    model = load_model(model, model_path)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Initialize evaluation metrics
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu_scores = []

    print("Evaluating...")
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model.generate(batch)  # Assuming `generate` method exists
            for idx, output in enumerate(outputs):
                # Decode predictions and original input
                predicted_text = tokenizer.decode(output)
                input_text = tokenizer.decode(batch[idx].cpu().numpy())

                print(f"Input: {input_text}")
                print(f"Predicted: {predicted_text}")

                # Calculate BLEU (compare input to predicted)
                reference = [input_text.split()]
                candidate = predicted_text.split()
                bleu = sentence_bleu(reference, candidate)
                bleu_scores.append(bleu)

                # Calculate ROUGE
                rouge_result = rouge.score(input_text, predicted_text)
                print(f"BLEU Score: {bleu:.4f}, ROUGE: {rouge_result}")

    # Average BLEU score
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"\nFinal Average BLEU Score: {avg_bleu:.4f}")

if __name__ == "__main__":
    model_path = "results/checkpoints/best_model.pt"  # Path to saved model
    test_data_path = "data/raw/test.txt"             # Path to test data
    tokenizer_path = "models/tokenizer.model"        # Path to trained tokenizer
    evaluate(model_path, test_data_path, tokenizer_path)
