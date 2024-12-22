import torch

try:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

        print(f"Properties of device 0: {torch.cuda.get_device_properties(0)}")
    else:
        print("CUDA is not available.")
except Exception as e:
    print(f"An error occurred: {e}")
