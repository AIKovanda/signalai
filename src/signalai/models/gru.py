import torch
from taskchain.parameter import AutoParameterObject
from torch import nn

from signalai.models.inceptiontime import InceptionBlock
from signalai.models.tools import get_activation


class GRU(AutoParameterObject, nn.Module):
    def __init__(self, input_size, hidden_size, layers, outputs):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.outputs = outputs
        self.rnn = nn.GRU(input_size, hidden_size, layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        # or:
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, outputs)

    def forward(self, x: torch.Tensor):
        # Set initial hidden states (and cell states for LSTM)
        x = torch.swapaxes(x, 1, 2)
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 28, 28), h0: (2, n, 128)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
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


class GRUMaxAvg(AutoParameterObject, nn.Module):
    def __init__(self, input_size, hidden_size, layers, outputs):
        super(GRUMaxAvg, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.outputs = outputs
        self.rnn = nn.GRU(input_size, hidden_size, layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, outputs)
        self.max = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x: torch.Tensor):
        x = torch.swapaxes(x, 1, 2)
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        swapped_out = torch.swapaxes(out, 1, 2)
        out1 = self.avg(swapped_out).view(-1, self.hidden_size)
        out2 = self.max(swapped_out).view(-1, self.hidden_size)
        return self.fc(torch.concat([out1, out2], dim=-1))


class InceptionGRU(AutoParameterObject, nn.Module):
    def __init__(self, input_size, hidden_size, layers, outputs, kernel_sizes, bias, combine=True, pool_size=1):
        assert hidden_size % 4 == 0
        super(InceptionGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.layers = layers
        self.pool_size = pool_size
        self.outputs = outputs
        self.combine = combine
        self.kernel_sizes = kernel_sizes
        n_filters = 32
        self.inc1 = InceptionBlock(
                in_channels=input_size,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=32,
                use_residual=False,
                use_batch_norm=False,
                activation=get_activation('tanh'),
                bias=bias,
            )
        if self.combine:
            self.inc2 = InceptionBlock(
                    in_channels=n_filters*4+input_size,
                    n_filters=int(hidden_size/4),
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=32,
                    use_residual=False,
                    use_batch_norm=False,
                    activation=get_activation('tanh'),
                    bias=bias,
                )
        self.rnn = nn.GRU(n_filters*4+input_size, hidden_size, layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*4*self.pool_size, outputs)
        self.max = nn.AdaptiveMaxPool1d(output_size=self.pool_size)
        self.avg = nn.AdaptiveAvgPool1d(output_size=self.pool_size)

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_size).to(x.device)
        x0 = self.inc1(x)
        x = torch.concat([x, x0], dim=1)
        out_rnn, _ = self.rnn(torch.swapaxes(x, 1, 2), h0)
        rnn_out_swapped = torch.swapaxes(out_rnn, 1, 2)
        outs = [self.avg(rnn_out_swapped).view(-1, self.hidden_size*self.pool_size),
                self.max(rnn_out_swapped).view(-1, self.hidden_size*self.pool_size)]
        if self.combine:
            out_inc = self.inc2(x)
            outs.append(self.avg(out_inc).view(-1, self.hidden_size*self.pool_size))
            outs.append(self.max(out_inc).view(-1, self.hidden_size*self.pool_size))
        return self.fc(torch.concat(outs, dim=-1))

    def dont_persist_default_value_args(self) -> list[str]:
        return ['pool_size']
