import torch
import torch.nn as nn

class Residual_1D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Residual_1D_Block, self).__init__()
        self.kernel_size = kernel_size
        self.cb1 = self.conv1d_block(in_channels, out_channels, dilation)
        self.cb2 = self.conv1d_block(out_channels, out_channels, dilation)
        
        if in_channels != out_channels:
            self.downsampling = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsampling = nn.Identity()

    def conv1d_block(self, in_channels, out_channels, dilation):
        left_padding = (self.kernel_size - 1) * dilation
        return nn.Sequential(
            nn.ZeroPad1d((left_padding, 0)),
            nn.Conv1d(in_channels, 
                    out_channels, 
                    self.kernel_size, 
                    dilation=dilation, 
                    padding=0),
            nn.BatchNorm1d(out_channels),
            # nn.LayerNorm([out_channels, 10000]),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        identity = self.downsampling(x)
        x = self.cb1(x)
        x = self.cb2(x)
        x += identity
        return x

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels_list, kernel_size, dilation_base=2):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        num_layers = len(out_channels_list)
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels_list[i-1]
            out_channels = out_channels_list[i]
            dilation = dilation_base ** i
            residual_block = Residual_1D_Block(in_channels, out_channels, kernel_size, dilation)
            self.layers.append(residual_block)

        self.lstm = nn.LSTM(input_size=out_channels_list[-1], hidden_size=256, 
                            num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 6)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        x = self.log_softmax(x)
        return x