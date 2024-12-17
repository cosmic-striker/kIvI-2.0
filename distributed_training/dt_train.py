import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from models.bart_model import BARTModel
from models.config import config
from utils.helper import save_model_checkpoint, load_model_checkpoint

# Dummy Dataset
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Initialize Distributed Environment
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Change this to server IP if needed
    os.environ['MASTER_PORT'] = '29500'      # Default port
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Distributed process initialized.")

# Clean up the Distributed Environment
def cleanup_distributed():
    dist.destroy_process_group()

# Training Function
def train(rank, world_size):
    print(f"[Rank {rank}] Starting training...")

    # Setup distributed environment
    setup_distributed(rank, world_size)

    # Configuration
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    batch_size = config['batch_size'] // world_size  # Divide batch size across workers

    # Create Model
    model = BARTModel(vocab_size=config['vocab_size']).to(device)
    model = DDP(model, device_ids=[rank])

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Dummy Data (Replace this with your actual dataset)
    data = torch.randint(0, config['vocab_size'], (1000, 50))  # Example input data
    targets = torch.randint(0, config['vocab_size'], (1000, 50))  # Example target data
    dataset = CustomDataset(data, targets)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Training Loop
    model.train()
    for epoch in range(config['epochs']):
        sampler.set_epoch(epoch)  # Shuffle data each epoch
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, config['vocab_size']), targets.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0 and rank == 0:
                print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if rank == 0:
            print(f"[Rank {rank}] Epoch {epoch} completed. Avg Loss: {running_loss / len(dataloader):.4f}")

        # Save checkpoint (only on rank 0)
        if rank == 0:
            save_model_checkpoint(model.module, optimizer, epoch, config['checkpoint_dir'])

    # Cleanup
    cleanup_distributed()
    print(f"[Rank {rank}] Training completed.")

# Entry Point for Multiprocessing
def main():
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2  # Default to 2 processes if no GPUs
    print(f"Using {world_size} processes for distributed training.")

    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()