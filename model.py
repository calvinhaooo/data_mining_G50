import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, classification=True):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.classification = classification

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # LSTM 层
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        out = self.relu(out)
        if self.classification:
            out = self.dropout(out)
        out = self.fc(out)

        out = self.relu(out)
        if self.classification:
            out = self.dropout(out)
        out = self.fc2(out)

        if self.classification:
            out = self.softmax(out)
        return out
