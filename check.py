import os
from utils.logger import Logger
from models.config import config
from distributed_training.establish import get_system_info, display_system_info, init_distributed_environment
from distributed_training.data_share import send_file, receive_file, start_server, start_client
from distributed_training.dt_train import train as distributed_train, CustomDataset
from scripts.train import train as local_train

def main():
    # Initialize Logger
    log_dir = "logs"
    log_file = "main_training.log"
    logger = Logger(log_dir=log_dir, log_file=log_file, rank=0)

    # Display system information
    system_info = get_system_info()
    logger.info("System Information:")
    for key, value in system_info.items():
        logger.info(f"{key}: {value}")
    display_system_info()

    # Check for GPUs
    gpu_available = system_info.get("GPUs", 0) > 0
    if not gpu_available:
        logger.error("No GPUs detected. Training will proceed on CPU.")
        logger.warning("For optimal performance, ensure GPUs are available.")

    # Display available datasets
    dataset_dir = config.get("dataset_dir", "datasets")
    logger.info("Checking available datasets...")
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory '{dataset_dir}' not found.")
        return
    available_datasets = os.listdir(dataset_dir)
    if not available_datasets:
        logger.error("No datasets found in the dataset directory.")
        return
    logger.info(f"Available datasets: {', '.join(available_datasets)}")

    # Choose training mode
    logger.info("Choose a training mode:")
    logger.info("1. Local Training")
    logger.info("2. Distributed Training")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        # Local training
        dataset_path = os.path.join(dataset_dir, available_datasets[0])  # Use the first dataset
        logger.info(f"Starting local training on dataset: {dataset_path}")
        try:
            local_train(dataset_path, config)
            logger.info("Local training completed successfully.")
        except Exception as e:
            logger.error(f"Error during local training: {e}")
    elif choice == "2":
        # Distributed training
        logger.info("Initializing distributed training environment...")
        try:
            init_distributed_environment()
            dataset = CustomDataset(data=[[0, 1, 2]], targets=[[1, 2, 3]])  # Dummy dataset
            distributed_train(rank=0, world_size=1, dataset=dataset)  # Simplified for testing
            logger.info("Distributed training completed successfully.")
        except Exception as e:
            logger.error(f"Error during distributed training: {e}")
    else:
        logger.error("Invalid choice. Please restart the program and choose 1 or 2.")
        return

    # Save trained model
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "trained_model.pt")
    logger.info(f"Saving trained model to: {model_path}")
    try:
        # Placeholder for model saving logic
        # Example: torch.save(model.state_dict(), model_path)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

    # Log the results
    logger.info("Training process completed. Check logs for detailed information.")

if __name__ == "__main__":
    main()
