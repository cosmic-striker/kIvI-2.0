import torch
from torch.utils.data import DataLoader, Dataset
from models.bart_model import BARTModel
from models.tokenizer import Tokenizer
from utils.helper import load_model
from rouge_score import rouge_scorer  # For ROUGE metrics
from nltk.translate.bleu_score import sentence_bleu  # For BLEU Score

# Custom Dataset for Evaluation
class TestDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        """
        Initializes the dataset by loading and tokenizing the data.
        Args:
            data_path (str): Path to the test data.
            tokenizer (Tokenizer): A tokenizer object to encode text.
        """
        self.tokenizer = tokenizer
        self.samples = self.load_data(data_path)

    def load_data(self, data_path):
        """
        Loads data from a text file.
        Args:
            data_path (str): Path to the test data file.
        Returns:
            list: A list of lines from the file.
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Gets the tokenized input IDs for a given index.
        Args:
            idx (int): Index of the data sample.
        Returns:
            torch.Tensor: Tokenized input IDs.
        """
        input_text = self.samples[idx]
        input_ids = self.tokenizer.encode(input_text)
        return torch.tensor(input_ids, dtype=torch.long)


# Evaluation Function
def evaluate(model_path, test_data_path, tokenizer_path, batch_size=8):
    """
    Evaluates the model on a test dataset using BLEU and ROUGE metrics.

    Args:
        model_path (str): Path to the saved model.
        test_data_path (str): Path to the test dataset.
        tokenizer_path (str): Path to the trained tokenizer.
        batch_size (int): Batch size for evaluation.
    """
    print("Starting evaluation...")

    # 1. Load tokenizer
    tokenizer = Tokenizer(tokenizer_path=tokenizer_path)
    print("Tokenizer loaded successfully.")

    # 2. Load test dataset
    dataset = TestDataset(test_data_path, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 3. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BARTModel()
    model = load_model(model, model_path)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 4. Initialize Metrics
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu_scores = []

    print("Evaluating...")
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            # Generate predictions
            outputs = model.generate(batch)  # Assuming `generate()` exists in BARTModel

            for idx, output in enumerate(outputs):
                # Decode predictions and input
                predicted_text = tokenizer.decode(output.cpu().numpy())
                input_text = tokenizer.decode(batch[idx].cpu().numpy())

                print(f"Input: {input_text}")
                print(f"Predicted: {predicted_text}")

                # Calculate BLEU
                reference = [input_text.split()]
                candidate = predicted_text.split()
                bleu = sentence_bleu(reference, candidate)
                bleu_scores.append(bleu)

                # Calculate ROUGE
                rouge_result = rouge.score(input_text, predicted_text)
                print(f"BLEU Score: {bleu:.4f}, ROUGE: {rouge_result}")

    # 5. Final Evaluation Results
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    print("\nEvaluation Complete.")
    print(f"Final Average BLEU Score: {avg_bleu:.4f}")


if __name__ == "__main__":
    # Paths for evaluation
    model_path = "results/checkpoints/best_model.pt"  # Path to saved model checkpoint
    test_data_path = "data/raw/test.txt"             # Path to test data
    tokenizer_path = "models/tokenizer.model"        # Path to trained tokenizer

    # Run evaluation
    evaluate(model_path, test_data_path, tokenizer_path)
