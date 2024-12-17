import torch
from torch.utils.data import DataLoader, Dataset
from models.bart_model import BARTModel
from models.tokenizer import Tokenizer
from models.config import config
from utils.helper import save_model, print_summary
from torch import optim, nn
import os

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data = self.load_data()

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.readlines()
        return [self.tokenizer.encode(line.strip()) for line in text]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

def train(model, tokenizer, train_data_path, epochs=5, batch_size=16, learning_rate=5e-5, device='cuda'):
    print("=== Starting Training ===")
    
    dataset = CustomDataset(train_data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")
        checkpoint_path = f"results/checkpoints/epoch_{epoch+1}.pt"
        save_model(model, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

    print("=== Training Complete ===")

def main():
    train_data_path = "data/raw/sample_train.txt"
    tokenizer = Tokenizer(train_data_path=train_data_path, vocab_size=config['vocab_size'])
    model = BARTModel(vocab_size=config['vocab_size'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(model, tokenizer, train_data_path, device=device)

if __name__ == "__main__":
    main()
