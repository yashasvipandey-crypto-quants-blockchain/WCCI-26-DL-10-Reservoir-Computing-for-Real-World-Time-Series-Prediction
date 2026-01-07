import torch
import torch.nn as nn

class OnlineLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super(OnlineLSTM, self).__init__()
        # 1. The LSTM Layer (The Memory)
        # It takes the sensor data and finds patterns over time
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # 2. The Linear Layer (The Predictor)
        # It takes the pattern found by LSTM and predicts the next number
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (Batch, Window_Size, Features)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # We only care about the very last step (the most recent memory)
        last_memory = lstm_out[:, -1, :]
        
        # Predict the future
        prediction = self.linear(last_memory)
        return prediction

def get_fuzzy_learning_rate(current_loss):
    """
    The 'Fuzzy Logic' Controller.
    Instead of a fixed learning speed, we adapt based on how wrong we are.
    """
    # If error is HUGE (Drift happened), learn fast!
    if current_loss > 0.5: 
        return 0.01 
    # If error is MEDIUM, learn normally.
    elif current_loss > 0.1: 
        return 0.001
    # If error is TINY (Stable), just fine-tune slowly.
    else: 
        return 0.0001