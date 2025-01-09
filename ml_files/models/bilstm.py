import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64):
        super(BiLSTMModel, self).__init__()
        self.bidirectional_lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.bidirectional_lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size1, batch_first=True, bidirectional=True)
        self.bidirectional_lstm3 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
        
        self.fc1 = nn.Linear(hidden_size2 * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.bidirectional_lstm1(x)
        x, _ = self.bidirectional_lstm2(x)
        x, _ = self.bidirectional_lstm3(x)
        
        # Take the last hidden state from the sequence
        x = x[:, -1, :]  # Shape: (batch_size, hidden_size2 * 2)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x