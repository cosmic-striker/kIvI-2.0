import os
import torch
import socket
from distributed_training.dt_train import main as distributed_train
from scripts.train import main as local_train
from scripts.evaluate import evaluate
from utils.logger import Logger
from models.config import config

def check_gpu_availability():
    """
    Checks if GPU is available. If not, prompts user for confirmation to continue with CPU.
    """
    if torch.cuda.is_available():
        logger.info("GPUs detected. Proceeding with GPU training.")
        return True
    else:
        logger.warning("No GPUs detected. Training will use CPU.")
        try:
            response = input("Do you want to continue with CPU training? (yes/no): ").strip().lower()
            if response == "yes":
                logger.info("User opted to proceed with CPU training.")
                return False
            else:
                logger.error("User aborted the training process due to lack of GPU.")
                exit(1)
        except KeyboardInterrupt:
            logger.error("User interrupted the process. Exiting.")
            exit(1)

def list_datasets(dataset_dir):
    """
    Lists available datasets in the given directory.
    """
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory '{dataset_dir}' does not exist.")
        exit(1)

    datasets = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    if not datasets:
        logger.error("No datasets found in the directory.")
        exit(1)

    logger.info("Available Datasets:")
    for idx, ds in enumerate(datasets):
        logger.info(f"{idx + 1}. {ds}")
    return datasets

def main():
    # Initialize logger
    global logger
    system_name = socket.gethostname()
    log_file = os.path.join("logs", "main_training.log")
    rank = int(os.getenv("RANK", 0))
    logger = Logger(log_file=log_file, rank=rank)

    logger.info("\n=== kIvI-2.0 Training Pipeline ===")
    logger.info(f"System Name: {system_name}")

    # GPU Check
    gpu_available = check_gpu_availability()

    # List Datasets
    dataset_dir = "datasets"  # Change this path as needed
    datasets = list_datasets(dataset_dir)

    logger.info("Please select a dataset to train on:")
    for idx, ds in enumerate(datasets):
        print(f"{idx + 1}. {ds}")

    selected_dataset = None
    while not selected_dataset:
        try:
            choice = int(input("Enter the number corresponding to the dataset: "))
            if 1 <= choice <= len(datasets):
                selected_dataset = datasets[choice - 1]
                logger.info(f"Selected Dataset: {selected_dataset}")
            else:
                logger.warning("Invalid selection. Please try again.")
        except ValueError:
            logger.warning("Please enter a valid number.")
        except KeyboardInterrupt:
            logger.error("User interrupted the dataset selection. Exiting.")
            exit(1)

    # Ensure Checkpoint Directory Exists
    os.makedirs(config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)

    # Choose Training Mode
    logger.info("Select training mode:")
    print("1. Local Training (Single System)")
    print("2. Distributed Training")

    while True:
        try:
            mode = input("Enter 1 or 2: ").strip()
            if mode == "1":
                logger.info("User selected Local Training mode.")
                logger.info("Starting Local Training...")
                try:
                    local_train(dataset_dir=os.path.join(dataset_dir, selected_dataset), gpu=gpu_available)
                except Exception as e:
                    logger.error(f"Error during Local Training: {e}")
                break
            elif mode == "2":
                logger.info("User selected Distributed Training mode.")
                logger.info("Starting Distributed Training...")
                try:
                    distributed_train(dataset_dir=os.path.join(dataset_dir, selected_dataset), gpu=gpu_available)
                except Exception as e:
                    logger.error(f"Error during Distributed Training: {e}")
                break
            else:
                logger.warning("Invalid selection. Please enter 1 or 2.")
        except KeyboardInterrupt:
            logger.error("User interrupted the training mode selection. Exiting.")
            exit(1)

    # Evaluate the Model
    logger.info("Starting Model Evaluation...")
    try:
        evaluate()
    except Exception as e:
        logger.error(f"Error during Model Evaluation: {e}")

    # Save Training Results
    logger.info("Saving the trained model...")
    try:
        trained_model_path = os.path.join(config['checkpoint_dir'], "final_trained_model.pth")
        # Simulate model saving logic (actual model saving code goes here)
        logger.info(f"Model saved at: {trained_model_path}")
    except Exception as e:
        logger.error(f"Error while saving the model: {e}")

    logger.info("Training and evaluation completed successfully.")

if __name__ == "__main__":
    main()
