import os
import torch
import socket
import psutil
from tabulate import tabulate
import torch.distributed as dist  

def get_system_info():
    """Get detailed system information including GPU and CPU."""
    system_name = socket.gethostname()

    # CPU information
    cpu_count = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    
    # Memory information (in GB)
    memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
    
    # GPU information
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_id = i
            gpu_info.append({"GPU ID": gpu_id, "GPU Name": gpu_name})

    system_info = {
        "System Name": system_name,
        "CPU Count (Logical)": cpu_count,
        "Physical Cores": physical_cores,
        "Total Memory (GB)": memory,
    }
    return system_info, gpu_info

def display_system_info():
    """Display system information and GPU details in a table format."""
    system_info, gpu_info = get_system_info()

    print("\n=== System Information ===")
    # Display basic system information in a table
    system_info_table = [
        ["System Name", system_info["System Name"]],
        ["CPU Count (Logical)", system_info["CPU Count (Logical)"]],
        ["Physical Cores", system_info["Physical Cores"]],
        ["Total Memory (GB)", system_info["Total Memory (GB)"]]
    ]
    print(tabulate(system_info_table, headers=["Parameter", "Value"], tablefmt="grid"))

    # Display GPU information if available
    if gpu_info:
        print("\n=== GPU Information ===")
        gpu_table = [[gpu["GPU ID"], gpu["GPU Name"]] for gpu in gpu_info]
        print(tabulate(gpu_table, headers=["GPU ID", "GPU Name"], tablefmt="grid"))
    else:
        print("\nNo GPUs detected. Proceeding with CPU training.")

def init_distributed_environment():
    """Initialize the distributed environment for multi-node training."""
    # Fetch the MASTER_ADDR and MASTER_PORT from environment or set defaults
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "12355")
    
    world_size = int(os.getenv("WORLD_SIZE", 1))  # Total number of processes/nodes
    rank = int(os.getenv("RANK", 0))  # Rank of the current process
    
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    
    # Initialize the process group for distributed training
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)  # Change "nccl" to "gloo"
    
    print("\n=== Distributed Training Setup ===")
    print(f"MASTER_ADDR: {master_addr}")
    print(f"MASTER_PORT: {master_port}")
    print(f"WORLD_SIZE: {world_size}")
    print(f"RANK: {rank}")
          
if __name__ == "__main__":
    display_system_info()
    init_distributed_environment()