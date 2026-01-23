import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional, kernel_size):  # bidirectional设为True则是BiLSTM
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)


    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return x

