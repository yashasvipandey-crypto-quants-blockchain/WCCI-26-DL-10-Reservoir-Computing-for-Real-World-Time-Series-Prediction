import pandas as pd
import numpy as np
import os

def get_industrial_stream(filepath="data/train_FD001.txt", window_size=10, sensor_idx=14):
    """
    Generator that yields (X_window, y_next) one by one.
    - If file exists: Reads C-MAPSS sensor data.
    - If file missing: Generates synthetic sine wave data (for testing).
    """
    
    # --- OPTION A: REAL DATA (C-MAPSS) ---
    if os.path.exists(filepath):
        print(f"Loading real data from {filepath}...")
        # C-MAPSS has 26 columns: Unit, Time, OpSet1-3, Sensor1-21
        cols = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=cols)
        
        # We focus on one specific sensor (e.g., Sensor 14 is sensitive to degradation)
        data = df[f's{sensor_idx}'].values
        
        # Normalize (Standardization is crucial for Neural Networks)
        mean_val = np.mean(data)
        std_val = np.std(data)
        data = (data - mean_val) / (std_val + 1e-6)

    # --- OPTION B: FAKE DATA (Fallback) ---
    else:
        print(f"Warning: {filepath} not found. Generating SYNTHETIC data for testing...")
        # Create a sine wave with some drift added later
        t = np.linspace(0, 100, 2000)
        data = np.sin(t) + np.random.normal(0, 0.05, 2000)
        
        # Add Concept Drift (Shift mean after 1000 steps)
        data[1000:] += 2.0 

    # --- STREAMING LOGIC ---
    # Yield one window at a time
    for i in range(len(data) - window_size):
        # Input: previous 10 readings
        X_window = data[i : i + window_size]
        # Target: the very next reading
        y_next = data[i + window_size]
        
        # Reshape for PyTorch (1 sample, Sequence Length, Features)
        # Returns numpy arrays, we convert to Tensor in the training loop
        yield X_window.reshape(1, window_size, 1), np.array([[y_next]])