import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from river import drift

# System path fix
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.data_loader import get_industrial_stream
from src.models.fuzzy_lstm import OnlineLSTM, get_fuzzy_learning_rate

def train_one_step(model, optimizer, criterion, X_tensor, y_tensor, use_fuzzy=False, current_loss=0.0):
    # 1. Predict
    model.eval()
    with torch.no_grad():
        pred = model(X_tensor)
    
    # 2. Calculate Loss for Reporting
    loss_val = criterion(pred, y_tensor).item()
    
    # 3. Adapt Learning Rate (The Secret Sauce)
    if use_fuzzy:
        lr = get_fuzzy_learning_rate(loss_val)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # 4. Train
    model.train()
    optimizer.zero_grad()
    train_pred = model(X_tensor)
    loss = criterion(train_pred, y_tensor)
    loss.backward()
    optimizer.step()
    
    return loss_val, pred.item()

def run_battle():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"STARTING AI BATTLE ON {device}")
    
    # --- CONTENDER 1: The Standard Model (Baseline) ---
    model_std = OnlineLSTM().to(device)
    optim_std = optim.Adam(model_std.parameters(), lr=0.001) # Fixed LR (Dumb)
    losses_std = []
    
    # --- CONTENDER 2: Your Model (Proposed Method) ---
    model_ours = OnlineLSTM().to(device)
    optim_ours = optim.Adam(model_ours.parameters(), lr=0.001) # Dynamic LR (Smart)
    losses_ours = []
    
    # Setup
    criterion = nn.MSELoss()
    stream = get_industrial_stream()
    drift_points = []
    drift_detector = drift.ADWIN()
    
    print("Simulating 2000 industrial cycles...")
    
    for t, (X, y) in enumerate(stream):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        
        # Train Standard
        l_std, _ = train_one_step(model_std, optim_std, criterion, X_tensor, y_tensor, use_fuzzy=False)
        losses_std.append(l_std)
        
        # Train Ours
        l_ours, _ = train_one_step(model_ours, optim_ours, criterion, X_tensor, y_tensor, use_fuzzy=True, current_loss=l_std)
        losses_ours.append(l_ours)
        
        # Detect Drift (using our model's error)
        drift_detector.update(l_ours)
        if drift_detector.drift_detected:
            drift_points.append(t)
            
        if t >= 2000: break

    # --- THE "MONEY SHOT" PLOT ---
    print("Generating Research Comparison Plot...")
    
    plt.figure(figsize=(14, 7))
    
    # Smoothing for cleaner visuals
    window = 30
    def smooth(data):
        return [sum(data[i:i+window])/window for i in range(len(data)-window)]

    plt.plot(smooth(losses_std), color='gray', linestyle='--', label='Standard LSTM (Baseline)', linewidth=1.5)
    plt.plot(smooth(losses_ours), color='red', label='Proposed Fuzzy-LSTM (Ours)', linewidth=2.5)
    
    # Highlight Drift
    for d in drift_points:
        plt.axvline(x=d, color='black', linestyle=':', alpha=0.5)
        plt.text(d, max(smooth(losses_std)), " Drift Event", rotation=90, verticalalignment='top')

    plt.title("Benchmarking Adaptation Speed: Standard vs. Proposed Method", fontsize=14)
    plt.xlabel("Time Steps (Online Stream)", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add a "Win" Box
    avg_std = np.mean(losses_std[1000:])
    avg_ours = np.mean(losses_ours[1000:])
    
    # Handle division by zero if error is 0
    if avg_std == 0: avg_std = 1e-6
    improvement = ((avg_std - avg_ours) / avg_std) * 100
    
    plt.figtext(0.15, 0.8, 
                f"RESULTS:\nStandard Error: {avg_std:.4f}\nOur Error: {avg_ours:.4f}\nImprovement: {improvement:.1f}%", 
                bbox=dict(facecolor='white', alpha=0.9))

    save_path = os.path.join(project_root, "paper_comparison_plot.png")
    plt.savefig(save_path)
    print(f"SUCCESS: Comparison graph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_battle()