import torch

print("--------------------------------------------------")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    print("SUCCESS: Your RTX 4060 is online and ready!")
else:
    print("WARNING: You are running on CPU (Slow).")
print("--------------------------------------------------")