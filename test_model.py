import torch
from src.models.fuzzy_lstm import OnlineLSTM

# 1. Create the model and move it to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

model = OnlineLSTM().to(device)
print(" Model created and moved to GPU!")

# 2. Create fake input data (1 sample, 10 time steps, 1 feature)
# We also move this data to the GPU
fake_input = torch.randn(1, 10, 1).to(device)

# 3. Ask the model to predict
output = model(fake_input)

print(f"Input shape: {fake_input.shape}")
print(f"Output value: {output.item()}")
print(" Success! The Brain is working.")