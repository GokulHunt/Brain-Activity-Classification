import torch
import torch.nn as nn

class Residual_1D_Block(nn.Module):
    def __init__(self, channels, kernel_size, downsampling):
        super(Residual_1D_Block, self).__init__()
        self.kernel_size = kernel_size
        self.cb1 = self.conv1d_block(in_channels=channels, out_channels=channels)
        self.cb2 = self.conv1d_block(in_channels=channels, out_channels=channels)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.downsampling = downsampling

    def conv1d_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, 
                    out_channels, 
                    self.kernel_size,
                    padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        identity = self.downsampling(x)
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.maxpool(x)
        x += identity
        return x

class CNN1d(nn.Module):
    def __init__(self, in_channels, kernel_list, res_layer=3, fixed_kernel=17):
        super(CNN1d, self).__init__()
        self.parrallel_cnn1d = nn.ModuleList()
        for kernel in kernel_list:
            self.parrallel_cnn1d.append(nn.Conv1d(in_channels, in_channels*4, 
                                                  kernel_size=kernel))
        self.out_channels = in_channels * 4
        self.batchnorm = nn.BatchNorm1d(self.out_channels)
        self.leakyrelu = nn.LeakyReLU()
        self.residual_blocks = nn.ModuleList()
        for res in range(res_layer):
            downsampling = nn.MaxPool1d(kernel_size=2, stride=2)
            self.residual_blocks.append(
                Residual_1D_Block(self.out_channels, fixed_kernel, downsampling))
        self.lstm = nn.LSTM(input_size=self.out_channels, hidden_size=256,
                            num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 6)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        outputs = []
        for cnn in self.parrallel_cnn1d:
            output = cnn(x)
            outputs.append(output)
        x = torch.cat(outputs, dim=2)
        print(x.shape)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        for rs in self.residual_blocks:
            x = rs(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        x = self.log_softmax(x)
        return x