# Import the function you just wrote
# Note: 'src.utils.data_loader' means src/utils/data_loader.py
from src.utils.data_loader import get_industrial_stream

print("--- TESTING DATA STREAM ---")

# Initialize the stream (It will generate fake sine-wave data if no file is found)
stream = get_industrial_stream()

# Get the first 3 data points
for i, (X, y) in enumerate(stream):
    print(f"\nStep {i+1}:")
    print(f"   Input (Sensor History): {X.shape}")  # Should be (1, 10, 1)
    print(f"   Target (Next Reading):  {y}")        # Should be a single number
    
    # Stop after 3 items so we don't print forever
    if i == 2:
        break

print("\n--- TEST COMPLETE ---")