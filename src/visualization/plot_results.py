import sys
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from river import drift

# Fix system path to find our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.utils.data_loader import get_industrial_stream
from src.models.fuzzy_lstm import OnlineLSTM, get_fuzzy_learning_rate

def run_and_plot():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OnlineLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    drift_detector = drift.ADWIN()
    stream = get_industrial_stream()
    
    # Storage for plotting
    losses = []
    drift_points = []
    
    print("Running simulation for plotting...")
    
    for t, (X, y) in enumerate(stream):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        
        # 1. Predict & Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_tensor)
        
        loss = criterion(y_pred, y_tensor)
        losses.append(loss.item())
        
        # 2. Detect Drift
        drift_detector.update(loss.item())
        if drift_detector.drift_detected:
            drift_points.append(t)
            
        # 3. Adapt (Train)
        lr = get_fuzzy_learning_rate(loss.item())
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_tensor)
        train_loss = criterion(train_pred, y_tensor)
        train_loss.backward()
        optimizer.step()
        
        if t >= 2000: break

    # --- PLOTTING ---
    print("Generating Plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Instantaneous Loss', alpha=0.5, color='blue')
    
    # Plot a moving average to make it look smoother
    window = 50
    if len(losses) > window:
        avg_losses = [sum(losses[i:i+window])/window for i in range(len(losses)-window)]
        plt.plot(range(window, len(losses)), avg_losses, label='Moving Avg Loss', color='black', linewidth=2)
    
    # Draw Red Lines for Drifts
    for d in drift_points:
        plt.axvline(x=d, color='red', linestyle='--', label='Drift Detected')
    
    plt.title("Online Deep Learning Performance under Concept Drift")
    plt.xlabel("Time Steps")
    plt.ylabel("Prediction Error (MSE)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    save_path = os.path.join(project_root, "results_graph.png")
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_and_plot()