import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # 添加的第一个线性层
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 添加的第二个线性层
        # self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # print(h0.shape, c0.shape)
        # print(c0)
        # LSTM 层
        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]
        out = self.fc1(out)
        # 应用激活函数（例如 ReLU）
        out = torch.relu(out)
        # 前向传播第二个线性层
        out = self.fc2(out)
        # print(out.shape)

        # 取出最后一个时间步的输出作为分类器的输入
        # out = self.fc(out[:, -1, :])

        # print(out)
        # x, _ = self.lstm(x)
        # x = self.fc(x)
        # out = x[-1, :]
        return out
