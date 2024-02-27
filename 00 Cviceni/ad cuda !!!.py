# Check for GPU
import torch
print(f"cuda je zde: {torch.cuda.is_available()}")

# Set device type
device = "cuda" if torch.cuda.is_available() else "cpu" # musi byt v "" !!!!
print(f"device je: {device}")

# Count number of devices
print(f"pocet cuda zarizeni je {torch.cuda.device_count()}")

# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu, tensor_on_gpu.device)


