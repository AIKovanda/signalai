import torch
from taskchain.parameter import AutoParameterObject
from torch import nn


class LSTM(AutoParameterObject, nn.Module):
    def __init__(self, input_size, hidden_size, layers, outputs):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.outputs = outputs
        self.rnn = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        # or:
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, outputs)

    def forward(self, x: torch.Tensor):
        # Set initial hidden states (and cell states for LSTM)
        x = torch.swapaxes(x, 1, 2)
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.layers, x.size(0), self.hidden_size).to(x.device)

        # x: (n, 28, 28), h0: (2, n, 128)

        # Forward propagate RNN
        out, _ = self.rnn(x, (h0, c0))
        # or:
        # out, _ = self.lstm(x, (h0,c0))

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)

        out = self.fc(out)
        # out: (n, 10)
        return out