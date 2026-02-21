import timm
import torch
from torch import nn

class Effe_CNN1d(nn.Module):
    def __init__(self, model, kernel_list, in_channels=16):
        super(Effe_CNN1d, self).__init__()
        self.parrallel_cnn1d = nn.ModuleList()
        for kernel in kernel_list:
            self.parrallel_cnn1d.append(nn.Conv1d(in_channels, in_channels*4, 
                                                  kernel_size=kernel, padding='same'))
        self.out_channels = in_channels * 4 * len(kernel_list)
        self.avgpool = nn.AvgPool1d(kernel_size=10, stride=10)
        self.batchnorm = nn.BatchNorm2d(1)
        self.leakyrelu = nn.LeakyReLU()
        self.model = timm.create_model(
            model,
            pretrained=True,
            num_classes=6,
            in_chans=1
        )
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 6),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        outputs = []
        for cnn in self.parrallel_cnn1d:
            output = cnn(x)
            outputs.append(output)
        x = torch.cat(outputs, dim=1)
        x = self.avgpool(x)
        x = x.unsqueeze(1)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        x = self.model(x)
        return x