import torch
import os

def save_model(model, save_path):
    """
    Save the model to the specified path.
    Args:
        model (torch.nn.Module): The model to save.
        save_path (str): Path to save the model checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully at: {save_path}")

def load_model(model, load_path, device=None):
    """
    Load the model weights from a checkpoint file.
    Args:
        model (torch.nn.Module): The model architecture to load weights into.
        load_path (str): Path to the model checkpoint.
        device (str): Device to map the model to ('cpu' or 'cuda').
    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at: {load_path}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"Model loaded successfully from: {load_path}")
    return model

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Save model and optimizer states as a checkpoint.
    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer state to save.
        epoch (int): Current epoch.
        loss (float): Loss value.
        save_path (str): Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at: {save_path}")

def load_checkpoint(model, optimizer, load_path, device=None):
    """
    Load model and optimizer states from a checkpoint.
    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer): Optimizer to load states into.
        load_path (str): Path to the checkpoint.
        device (str): Device to map the model to ('cpu' or 'cuda').
    Returns:
        tuple: Loaded model, optimizer, epoch, and loss.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at: {load_path}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from: {load_path}")
    return model, optimizer, epoch, loss

def print_summary(model):
    """
    Print a summary of the model architecture.
    Args:
        model (torch.nn.Module): The model to summarize.
    """
    print("Model Summary:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
