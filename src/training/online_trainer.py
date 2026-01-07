import sys
import os

# --- THE FIX: Tell Python where the project root is ---
# Get the folder where this script is saved
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to find the main project folder
project_root = os.path.dirname(os.path.dirname(current_dir))
# Add it to the system path
sys.path.append(project_root)
# -----------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from river import drift

# Import our own custom tools
from src.utils.data_loader import get_industrial_stream
from src.models.fuzzy_lstm import OnlineLSTM, get_fuzzy_learning_rate

def run_online_training():
    # 1. SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Online Training on {device} ---")
    
    # Initialize Model
    model = OnlineLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Initialize Concept Drift Detector (ADWIN is standard for this)
    drift_detector = drift.ADWIN()
    
    # Get Data Stream
    stream = get_industrial_stream()
    
    # Variables to track performance
    total_loss = 0
    drift_count = 0
    
    # 2. THE LOOP (Test-Then-Train)
    print("Stream started. Processing data points...")
    
    for t, (X, y) in enumerate(stream):
        
        # Prepare Data (Move to GPU)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        
        # --- A. PREDICT (Test phase) ---
        model.eval()
        with torch.no_grad():
            y_pred = model(X_tensor)
        
        # --- B. EVALUATE ---
        loss = criterion(y_pred, y_tensor)
        current_loss = loss.item()
        total_loss += current_loss
        
        # Check for Drift
        drift_detector.update(current_loss)
        if drift_detector.drift_detected:
            drift_count += 1
            print(f"!! DRIFT DETECTED at Step {t} !! Adjusting Model...")
            # (Optional: You could reset the optimizer here if drift is severe)
            
        # --- C. ADAPT (Train phase) ---
        # 1. Get Fuzzy Learning Rate
        lr = get_fuzzy_learning_rate(current_loss)
        
        # 2. Update Optimizer with new LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 3. Backpropagation
        model.train()
        optimizer.zero_grad()
        
        # Forward pass again to build the gradient graph
        train_pred = model(X_tensor)
        train_loss = criterion(train_pred, y_tensor)
        
        train_loss.backward()
        optimizer.step()
        
        # --- LOGGING ---
        if t % 200 == 0:
            avg_loss = total_loss / (t + 1)
            print(f"Step {t} | Loss: {current_loss:.4f} | Avg Loss: {avg_loss:.4f} | LR: {lr} | Drifts: {drift_count}")
            
        # Stop after 2000 steps for this demo
        if t >= 2000:
            break

    print("--- Training Complete ---")

if __name__ == "__main__":
    run_online_training()