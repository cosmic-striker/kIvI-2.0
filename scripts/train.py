import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.config import config
from utils.logger import Logger

# Dummy model and dataset for demonstration
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size, input_dim):
        self.size = size
        self.input_dim = input_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random input and output tensors for dummy data
        x = torch.randn(self.input_dim)
        y = torch.randint(0, 2, (1,)).item()
        return x, y

def train(dataset_dir, gpu):
    """
    Function to train the model locally.
    
    Args:
    - dataset_dir (str): Path to the dataset directory.
    - gpu (bool): Whether to use GPU or CPU for training.
    """
    # Initialize logger
    rank = int(os.getenv("RANK", 0))
    log_file = os.path.join(config.get("log_dir", "logs"), "local_training.log")
    logger = Logger(log_file=log_file, rank=rank)

    logger.info("=== Starting Local Training ===")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Using GPU: {gpu}")

    # Set device
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Dataset and DataLoader
    input_dim = config.get("input_dim", 10)  # Replace with actual input dim
    dataset_size = config.get("dataset_size", 1000)  # Replace with actual size
    batch_size = config.get("batch_size", 32)
    num_epochs = config.get("num_epochs", 10)
    
    dataset = DummyDataset(size=dataset_size, input_dim=input_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Dataset size: {len(dataset)} | Batch size: {batch_size} | Epochs: {num_epochs}")

    # Model, Loss, Optimizer
    model = SimpleModel(input_size=input_dim, output_size=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        logger.info(f"Epoch {epoch+1} completed. Average Loss: {running_loss/len(dataloader):.4f}")

    # Save Model
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "local_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved at: {model_path}")
    logger.info("=== Local Training Completed ===")
