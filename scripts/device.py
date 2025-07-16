import torch

print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print("CUDA available:", True)
    print("Current device index:", torch.cuda.current_device())
    print("GPU count:", torch.cuda.device_count())
    print("First GPU name:", torch.cuda.get_device_name(0))
    print("First GPU compute capability:", torch.cuda.get_device_capability(0))
else:
    print("CUDA not available. GPU not detected or not accessible.")
